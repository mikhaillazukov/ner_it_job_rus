import sys
import os

from traitlets import default
import gpt4free.quora as quora
import pandas as pd
from tqdm import tqdm
import time
import click


def generate_completion(prompt, model, token):
    response = quora.Completion.create(model=model, prompt=prompt, token=token)
    print(response.choices[0].text)
    return response.choices[0].text


@click.command()
@click.option('--path_to_csv', '-p', type=click.Path(exists=True))
@click.option('--token', '-t', type=str)
@click.option("--start", '-s', default=0, type=int)
@click.option("--model", '-m', type=str)
def do_all(path_to_csv, token, start, model):
    # Choose a model and type your prompt

    df = pd.read_csv(path_to_csv, index_col=0)
    df["result"] = ""

    i = start
    total = len(df)
    while (i < total):
        print(i)
        try:
            text = df['text'][i]
            if len(text) > 2000:
                i += 1
                continue
            result = generate_completion(
                text + '\n Извлеки из текста сущности TITLE, COMPANY, FORMAT, SALARY, HARDSKILLS, SOFTSKILLS и выведи их в следующем формате: СУЩНОСТЬ: СПИСОК ОТВЕТОВ. Больше ничего другого выводить не надо',
                model, token)
            df['result'][i] = result
            df.to_csv(path_to_csv, sep=",")
            time.sleep(20)
            i += 1
        except RuntimeError:
            print("Упал :(")
            time.sleep(100)


if __name__ == "__main__":
    do_all()
