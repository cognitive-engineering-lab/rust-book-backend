import pandas as pd
from pathlib import Path
from datetime import datetime
import subprocess as sp
import ujson
import sqlite3
import zlib

import rs_utils


QUIZ_DIR = Path('~/rust-book/quizzes').expanduser()
LOG_PATH = Path('../server/log.sqlite').resolve()


def date_for_commit(commit_hash):
    date_str = sp.check_output(['git', 'show', '-s', '--format=%ci', commit_hash], cwd=QUIZ_DIR).decode('utf-8').strip()
    return datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S %z')            


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
            correct = not row['answer']['doesCompile']
        else:
            correct = row['correct']
        correct_v2.append(correct)
    answers_flat['correct_v2'] = correct_v2


def load_latest_answers():
    print('Loading answers...')
    answers = load_log('answers')
    answers['commitHash'] = answers.commitHash.map(lambda s: s.strip())

    # Load in all quiz data and get version metadata
    print('Loading quizzes...')
    quiz_params = [(row.quizName, row.commitHash) for _, row in answers.iterrows()]
    quizzes = rs_utils.Quizzes(quiz_params)

    # Convert hashes to version numbers
    print("Postprocessing data...")
    answers['version'] = answers.apply(lambda row: quizzes.version(row.quizName, row.commitHash), axis=1)

    # Convert UTC timestamp to datetime
    answers['timestamp'] = pd.to_datetime(answers['timestamp'], unit='ms')

    # Only keep the first attempt
    answers = answers[answers.attempt == 0]

    # Remove example data
    answers = answers[answers.quizName != 'example-quiz']

    # Only keep the latest complete answer for a given user/quiz pair
    get_latest = lambda group: group.iloc[group.timestamp.argmax()]
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

    print('Loaded!')

    return answers, answers_flat, quizzes
