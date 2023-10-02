import pandas as pd
from pathlib import Path
from datetime import datetime, date
import subprocess as sp
import ujson
import sqlite3
import zlib
import shlex
import numpy as np
import os
import textwrap

import rs_utils

QUIZ_DIR = Path('../code/rust-book/quizzes').expanduser()
LOG_PATH = Path('../data/log.sqlite').resolve()

def run_cmd(args):
    return sp.check_output(shlex.split(args), cwd=QUIZ_DIR).decode('utf-8').strip()

def date_for_commit(commit_hash):
    date_str = run_cmd(f'git show -s --format=%ci {commit_hash}')
    return datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S %z')            

def date_for_tag(tag):
    return date_for_commit(run_cmd(f'git rev-list -n 1 {tag}'))

def load_log(name):
    logs = sqlite3.connect(f'file:{LOG_PATH}?mode=ro', uri=True)

    def load(row):
        (data_bytes,) = row
        data_str = zlib.decompress(data_bytes)
        obj = ujson.loads(data_str)
        payload = obj.pop('payload')
        return {**obj, **payload}

    cursor = logs.execute(f'SELECT data FROM {name}')
    rows = [load(row) for row in cursor]
    return pd.DataFrame(rows)


def patch_tracing_answers(answers_flat, quizzes):
    correct_v2 = []
    for _, row in answers_flat.iterrows():
        quiz = quizzes.schema(row.quizName, row.commitHash)
        question = quiz['questions'][row['question']]
        if question['type'] == 'Tracing' and not question['answer']['doesCompile']:
            try:
                correct = not row['answer']['doesCompile']
            except Exception:
                correct = True
        else:
            correct = row['correct']
        correct_v2.append(correct)
    answers_flat['correct_v2'] = correct_v2

def patch_incorrect_question_ids(answers_flat):
    df = answers_flat
    fixes = [
        {
            'before-id': 'd070eb9e-4527-453a-8c9b-698739a3dd6a',
            'before': [
                {
                    'quiz': 'ch10-04-inventory',
                    'question': 2,
                    'end-time': date(2023, 1, 18)
                },
                {
                    'quiz': 'ch10-04-inventory',
                    'question': 5,
                    'end-time': date(2023, 3, 2)
                }
            ],
            'after-id': 'aa93c497-9864-4799-b69b-7de42c158f2a',        
        },
    #     {
    #         'before-id': '404b2cf7-7d29-45d7-86c4-7c806a071d7b',
    #         'before-quiz': 'ch17-04-inventory',
    #         'before-question': 5,
    #         'after-id': '2da21d0e-5722-4908-b528-dc87bbce1faf',
    #     }
    ]

    for fix in fixes:
        for pos_at_time in reversed(fix['before']):        
            before = df[(df.timestamp.dt.date <= pos_at_time['end-time']) & (df.id == fix['before-id'])]
            before = before[(before.quizName == pos_at_time['quiz']) & (before.question == pos_at_time['question'])]
            answers_flat.loc[before.index, 'id'] = fix['after-id']

def add_fractional_score(answers_flat, quizzes):
    frac_score = []
    for _, row in answers_flat.iterrows():
        quiz = quizzes.schema(row.quizName, row.commitHash)
        question = quiz['questions'][row['question']]
        if question['type'] == 'MultipleChoice' and type(question['answer']['answer']) is list:
            score = 0.            
        else:
            score = row['correct_v2']
        frac_score.append(score)
    answers_flat['frac_score'] = score


def load_latest_answers():
    if os.environ.get('ANALYSIS_ENV') != 'docker':
        sp.check_call(['git', 'pull'], cwd=QUIZ_DIR)

    print('Loading answers...')
    answers = load_log('answers')
    answers['commitHash'] = answers.commitHash.map(lambda s: s.strip())

    # Load in all quiz data and get version metadata
    print('Loading quizzes...')
    quiz_params = [(row.quizName, row.commitHash) for _, row in answers.iterrows()]

    quizzes = rs_utils.Quizzes(quiz_params)
    
    # Filter out any inputs ignored during quiz processing (see rs_utils.rs for why)    
    print('Ignored inputs: ', len(quizzes.ignored_inputs()))
    answers = answers.loc[answers.index.difference(quizzes.ignored_inputs())]
    
    # Convert hashes to version numbers
    print("Postprocessing data...")
    answers['version'] = answers.apply(lambda row: quizzes.version(row.quizName, row.commitHash), axis=1)

    # Convert UTC timestamp to datetime
    answers['timestamp'] = pd.to_datetime(answers['timestamp'], unit='ms')

    # Only keep the first attempt
    answers = answers[answers.attempt == 0]

    # Remove example data
    answers = answers[answers.quizName != 'example-quiz']

    # Only keep the first complete answer for a given user/quiz pair
    get_latest = lambda group: group.iloc[group.timestamp.argmin()]
    did_complete_quiz = lambda row: len(row.answers) == len(quizzes.schema(row.quizName, row.commitHash)['questions'])
    groups = ['sessionId', 'quizName', 'quizHash']
    answers = answers \
        .loc[lambda df: df.apply(did_complete_quiz, axis=1)] \
        .groupby(groups) \
        .apply(get_latest) \
        .drop(columns=groups) \
        .reset_index()

    answers['frac_correct'] = answers.answers.map(lambda a: len([o for o in a if o['correct']]) / len(a))

    answers_flat = []
    for _, response in answers.iterrows():
        quiz = quizzes.schema(response.quizName, response.commitHash)
        for i, ans in enumerate(response.answers):
            explanation = ans['answer'].pop('explanation', ans.pop('explanation', None))
            row = {**response, **ans, 'explanation': explanation, 'question': i, 'id': quiz['questions'][i].get('id', None)}
            del row['answers']
            answers_flat.append(row)

    answers_flat = pd.DataFrame(answers_flat)
    patch_tracing_answers(answers_flat, quizzes)
    patch_incorrect_question_ids(answers_flat)

    print('Loaded!')

    return answers, answers_flat, quizzes

def cohens_d(x1, x2):
    n1, n2 = len(x1), len(x2)
    sd_pooled = np.sqrt(((n1-1) * np.std(x1)**2 + (n2-1) * np.std(x2)**2) / (n1+n2-2))
    return (x2.mean() - x1.mean()) / sd_pooled

def format_pct(n):
    return f'{n*100:.0f}\\%'

def format_effect(row):
    pct = format_pct(row.effect)
    if row.effect > 0:
        pct = f'+{pct}'    
    return pct

def format_p(p, alpha):
    if p < 0.001:
        s = '<0.001'
    else:
        s = f'{p:.03f}'
    if p < alpha:
        s = '\textbf{'+s+'}'
    return s

def fmt_schema(s):
    print(f'QUESTION TYPE: {s["type"]}')
    print('PROMPT:')
    if s['type'] == 'Tracing':
        print(s['prompt']['program'])
        if s['answer']['doesCompile']:
            print('ANSWER: DOES compile, stdout is:\n' + textwrap.indent(s['answer']['stdout'], '  '))
        else:
            print('ANSWER: does NOT compile')
    else:
        print(s['prompt']['prompt'])
        if s['type'] == 'MultipleChoice':
            ans = s['answer']['answer']
            if type(ans) == str:
                ans = [ans]
            for a in ans:
                print(f'✓ {a}')
            for d in s['prompt']['distractors']:
                print(f'✗ {d}')
        else:
            print('ANSWER: ' + s['answer']['answer'])

def print_sep():
    print('\n' + '=' * 20 + '\n')

# from pympler import asizeof
# if __name__ == "__main__":
#     answers, answers_flat, quizzes = load_latest_answers()
#     print('answers', asizeof.asizeof(answers))
#     print('answers_flat', asizeof.asizeof(answers_flat))
#     print('quizzes', asizeof.asizeof(quizzes))

