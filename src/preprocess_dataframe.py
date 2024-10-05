import html
import re
from html import unescape as html_unescape

import pandas as pd
import tqdm
from lingua import Language, LanguageDetectorBuilder
from nltk.tokenize import TweetTokenizer
from string import punctuation

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

tqdm.tqdm.pandas()


def clean_dataframe(df, verbose=True):

    # Assert that df has `user_id` and `tweet` columns
    assert "user_id" in df.columns, "DataFrame must have a 'user_id' column"
    assert "tweet" in df.columns, "DataFrame must have a 'tweet' column"

    task_counter = 0
    total_count = len(df)

    def log_remove(task, after):
        if not verbose:
            return

        nonlocal task_counter
        nonlocal total_count

        task_counter += 1
        print(
            f"    {task_counter}. Removed {total_count - after} {task} (total = {after})"
        )
        total_count = after

    def log(text, numbered=True):
        if not verbose:
            return

        if numbered:
            nonlocal task_counter
            task_counter += 1
            print(f"    {task_counter}. {text}")
        else:
            print(text)

    before_n = len(df)
    log(f"Number of tweets: {before_n} before Processing", numbered=False)

    # Start processing tweets:
    log("Start processing tweets:", numbered=False)

    # --------------------------
    # --------------------------
    # --------------------------
    # Step 1: Remove unwanted tweets

    # Drop empty tweets
    df = df.dropna(subset=["tweet"])
    df = df[df["tweet"] != " "]
    df = df[df["tweet"] != ""]
    log_remove("empty tweets", after=len(df))

    # Remove duplicates
    df = df.drop_duplicates(subset=["user_id", "tweet"], keep="first")
    log_remove("duplicates tweets", after=len(df))

    # Remove tweets that start with @
    df["tweet"] = df["tweet"].str.strip()  # trim the tweets
    df = df[~df["tweet"].str.startswith("@")]
    log_remove("tweets that start with @", after=len(df))

    # 5. Remove tweets that star with 'Wordle'
    df = df[~df["tweet"].str.startswith("Wordle")]
    df = df[~df["tweet"].str.startswith("wordle")]
    log_remove("tweets that start with 'Wordle'", after=len(df))

    # --------------------------
    # --------------------------
    # --------------------------
    # Step 2: Clean the tweets

    # Remove URLs
    def remove_urls(text):
        # Define the regex pattern for URLs
        url_pattern = r"https?://\S+|www\.\S+"
        # Replace URLs with an empty string
        return re.sub(url_pattern, "", text)

    log("Removing URLs:")
    df["tweet"] = df["tweet"].progress_apply(remove_urls)

    # Remove HTML entities
    log("Removing HTML entities:")
    df["tweet"] = df["tweet"].progress_apply(html_unescape)

    # Replace `'` with space
    log("Replacing ' with space:")
    df["tweet"] = df["tweet"].str.replace("'", " ")

    # Remove tweets with less than 5 words
    def is_langer_then(tokens, n) -> bool:
        count = 0
        for t in tokens:
            if t.startswith("#") or t.startswith("@"):
                continue
            if not t[0].isalpha():
                continue
            count += 1

        return count >= n

    tknzr = TweetTokenizer()

    log("Tokenizing tweets")
    df["tokens"] = df["tweet"].progress_apply(tknzr.tokenize)
    df = df[df["tokens"].progress_apply(lambda x: is_langer_then(x, 5))]
    log_remove("tweets with less than 5 words", after=len(df))
    df = df.drop(columns=["tokens"])  # remove tokens column

    # Remove non english languages
    log("    * Removing non english languages", numbered=False)
    languages = [Language.ENGLISH, Language.FRENCH, Language.GERMAN, Language.SPANISH]
    detector = LanguageDetectorBuilder.from_languages(*languages).build()

    tweets = df["tweet"].tolist()
    languages = detector.detect_languages_in_parallel_of(tweets)
    languages = [l.iso_code_639_1.name if l is not None else "None" for l in languages]
    df["language"] = languages
    df = df[df["language"] == "EN"]

    log_remove("non english languages", after=len(df))

    # remove language column
    df = df.drop(columns=["language"])

    # Remove non ASCII characters
    # from unidecode import unidecode
    # This will remove emojis and other langues characters
    # df['tweet'] = df['tweet'].progress_apply(unidecode)
    # print("    8. Removed non ASCII characters")

    # Remove white space
    def remove_whitespace(text):
        # Remove leading and trailing white space
        text = text.strip()
        # Remove multiple white spaces
        text = re.sub("\s+", " ", text)
        return text

    log("Removing white space:")
    df["tweet"] = df["tweet"].progress_apply(remove_whitespace)

    # Remove numbers
    def remove_numbers(text):
        return re.sub(r"\d+", "", text)

    log("Removing numbers:")
    df["tweet"] = df["tweet"].progress_apply(remove_numbers)

    # 10. Again remove empty tweets and less than 5 words
    df = df.dropna(subset=["tweet"])
    df = df[df["tweet"] != " "]
    df = df[df["tweet"] != ""]
    df = df[df["tweet"].progress_apply(lambda x: is_langer_then(x, 5))]

    log_remove("empty tweets and less than 5 words", after=len(df))

    # 11. Remove HTML entities again
    log("Removing HTML entities again:")
    df["tweet"] = df["tweet"].progress_apply(html_unescape)

    # 2. Remove duplicates
    df = df.drop_duplicates(subset=["user_id", "tweet"], keep="first")

    log_remove("duplicates tweets", after=len(df))

    log(f"Number of tweets: {len(df)} after processing", numbered=False)

    return df


def process_tokens(tokens_lists: list[list[str]]) -> list[str]:

    stop_words = set(stopwords.words("english"))
    # stop_words.add("i'm")
    # stop_words.add("i've")
    # stop_words.add("i'll") # TODO

    out = []

    for tokens in tokens_lists:
        # tokens is list of str

        # Remove stopwords and lower case
        words = [w.lower() for w in tokens if w.lower() not in stop_words]

        # Remove punctuation
        words = [w for w in words if w not in punctuation]

        # remove numbers
        words = [w for w in words if not w.isnumeric()]

        # Lemmatize
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(w, pos="n") for w in words]
        words = [lemmatizer.lemmatize(w, pos="v") for w in words]
        words = [lemmatizer.lemmatize(w, pos="a") for w in words]
        words = [lemmatizer.lemmatize(w, pos="r") for w in words]

        t = " ".join(words)

        out.append(t)

    return out
