import pandas as pd


def prepare_train():
    df = pd.read_parquet('./data/synthetic_text_to_sql/train.snappy.parquet')

    df['prompt'] = df['sql_prompt'] + " with given SQL schema " + df['sql_context']
    df['completion'] = "SQL: " + df['sql'] + " Explanation: " + df['sql_explanation']
    df = df[['prompt', 'completion']]

    print(df.head(10))

    # Convert the DataFrame to a JSON format, with each record on a new line
    # save as .jsonl
    df.to_json('train.jsonl', orient='records', lines=True)


def prepare_test_valid():
    df = pd.read_parquet('./data/synthetic_text_to_sql/test.snappy.parquet')

    df['prompt'] = df['sql_prompt'] + " with given SQL schema " + df['sql_context']
    df['completion'] = "SQL: " + df['sql'] + "Explanation: " + df['sql_explanation']
    df = df[['prompt', 'completion']]

    # Calculate split index for two-thirds
    split_index = int(len(df) * 2 / 3)

    # Split the DataFrame into two parts
    test_df = df[:split_index]
    valid_df = df[split_index:]

    print(test_df.head(10))
    print(valid_df.head(10))

    # Save the subsets to their respective JSONL files
    test_df.to_json('test.jsonl', orient='records', lines=True)
    valid_df.to_json('valid.jsonl', orient='records', lines=True)


prepare_train()
prepare_test_valid()
