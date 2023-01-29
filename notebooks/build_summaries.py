import pandas as pd
import json
from statsmodels.stats.proportion import proportion_confint

from utils import load_latest_answers


def main():
    answers, answers_flat, quizzes = load_latest_answers()
    def summarize(df, groups, metric, ci_estimator, extra):
        g = df.groupby(groups)
        def ci(df):
            ser = df[metric]
            mu = ser.mean()
            lower, upper = ci_estimator(ser)
            return pd.DataFrame([{
                'mean': mu,
                'lower': lower,
                'upper': upper,
                'N': len(ser),
                **{k: df[~df[metric]][k] for k in extra}
            }])
        df2 = g.apply(ci).reset_index().drop(columns=[f'level_{len(groups)}'])
        return df2.groupby(list(set(groups) - {'version'})).apply(lambda df: df.iloc[df.version.argmax()]).reset_index(drop=True)
 
    def normal_estimator(ser):
        mu = ser.mean()
        sigma = ser.sem()
        return (mu - sigma * 2, mu + sigma * 2)

    quizSummary = summarize(
        answers, 
        ['quizName', 'version'], 
        'frac_correct', 
        normal_estimator,
        [])

    questionSummary = summarize(
        answers_flat, 
        ['quizName', 'version', 'question'], 
        'correct_v2', 
        lambda ser: proportion_confint(ser.sum(), len(ser)),
        ['answer'])

    explanations = answers_flat[answers_flat.explanation.notnull()] \
        .groupby(['quizName', 'version', 'question']) \
        .apply(lambda group: group.explanation.tolist()) \
        .rename("explanations") \
        .reset_index()

    quizSummary.to_json('data/quiz-summary.json', orient="records")
    questionSummary.to_json('data/question-summary.json', orient="records")
    explanations.to_json('data/explanations.json', orient="records")
    quiz_clean = {}
    for _, row in answers.iterrows():
        versions = quiz_clean.setdefault(row.quizName, {})
        versions.setdefault(
            quizzes.version(row.quizName, row.commitHash), 
            quizzes.schema(row.quizName, row.commitHash))
    json.dump(quiz_clean, open('data/quiz-schemas.json', 'w'))


if __name__ == "__main__":
    main()
