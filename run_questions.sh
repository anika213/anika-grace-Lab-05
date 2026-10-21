#!/bin/sh
# Usage: sh run_questions.sh [DATA_DIR [NUM_ARTICLES]]
# Optional environment variables: PYTHON (default python3), OUTPUT_DIR.
# Each command trains its model again, matching the existing Python CLIs.

set -eu

if [ "$#" -gt 2 ]; then
    printf 'Usage: %s [DATA_DIR [NUM_ARTICLES]]\n' "$0" >&2
    exit 2
fi

data_dir=${1:-/courses/cs159/data/wikipedia/enwiki_sampled/}
if [ ! -d "$data_dir" ]; then
    printf 'Corpus directory does not exist: %s\n' "$data_dir" >&2
    exit 2
fi
if [ "$#" -eq 2 ]; then
    case $2 in
        ''|*[!0-9]*|0) printf 'NUM_ARTICLES must be a positive integer.\n' >&2; exit 2 ;;
    esac
fi

script_dir=$(CDPATH='' cd -P -- "$(dirname -- "$0")" && pwd)
python=${PYTHON:-python3}
output_dir=${OUTPUT_DIR:-"$PWD/results-$(date +%Y%m%d-%H%M%S)-$$"}
mkdir -p "$output_dir"

# Start a fresh file for each question; its queries append below.
for question in Q2 Q3 Q4 Q6 Q7A Q8A Q7B Q8B
do
    : > "$output_dir/$question.txt"
done

# Preserve the optional article limit as separate arguments, without eval.
if [ "$#" -eq 2 ]; then
    set -- -n "$2"
else
    set --
fi

failures=0
run_number=0
run() {
    section=$1
    label=$2
    shift 2
    run_number=$((run_number + 1))
    log_file="$output_dir/${section}.txt"
    printf '\n[%s] %s\nOutput: %s\n' "$section" "$label" "$log_file"
    printf '\n[%s] %s\n\n' "$section" "$label" >> "$log_file"
    if "$python" -u "$@" >> "$log_file" 2>&1; then
        printf 'Saved to %s\n' "$log_file"
    else
        status=$?
        printf 'FAILED (exit %s)\n' "$status" >> "$log_file"
        printf 'FAILED (exit %s): %s — %s\n' "$status" "$section" "$label" >&2
        failures=$((failures + 1))
    fi
}

# # Q2: CountDocumentVectors (implemented using TF-IDF).
# for query in \
#     'Nikola Tesla birthday' \
#     'Ingredients in Fig Newtons' \
#     'Forms of communication during World War 2' \
#     'The nature of God'
# do
#     run Q2 "$query" "$script_dir/wiki_search.py" "$data_dir" -m tfidf -q "$query" "$@"
# done

# # Q3: SVDDocumentVectors.
# for query in \
#     'liberal arts college in southern california' \
#     'first manned mission to the moon' \
#     'marvel movie about an insect themed superhero'
# do
#     run Q3 "$query" "$script_dir/wiki_search.py" "$data_dir" -m svd -q "$query" "$@"
# done

# # Q4: SVDDocumentVectors.
# for query in \
#     'Nikola Tesla birthday' \
#     'Ingredients in Fig Newtons' \
#     'Forms of communication during World War 2' \
#     'The nature of God'
# do
#     run Q4 "$query" "$script_dir/wiki_search.py" "$data_dir" -m svd -q "$query" "$@"
# done

# Q6: SVDWordVectors (the default); each comparison pair is adjacent.
for word in male female black white africa europe
do
    run Q6 "$word" "$script_dir/wiki_word_lookup.py" "$data_dir" -w "$word" "$@"
done

# # Q7A: SentenceTransformerDocumentVectors.
# for query in \
#     'liberal arts college in southern california' \
#     'first manned mission to the moon' \
#     'marvel movie about an insect themed superhero'
# do
#     run Q7A "$query" "$script_dir/wiki_search.py" "$data_dir" -m sent -q "$query" "$@"
# done

# # Q8A: SentenceTransformerDocumentVectors.
# for query in \
#     'Nikola Tesla birthday' \
#     'Ingredients in Fig Newtons' \
#     'Forms of communication during World War 2' \
#     'The nature of God'
# do
#     run Q8A "$query" "$script_dir/wiki_search.py" "$data_dir" -m sent -q "$query" "$@"
# done

# Q7B: Word2Vectors.
for word in college science
do
    run Q7B "$word" "$script_dir/wiki_word_lookup.py" "$data_dir" -v -w "$word" "$@"
done

# Final requested comparison, labeled Q8B: Word2Vectors.
for word in male female black white africa europe
do
    run Q8B "$word" "$script_dir/wiki_word_lookup.py" "$data_dir" -v -w "$word" "$@"
done

printf '\nFinished %s searches; %s failed. Results: %s\n' "$run_number" "$failures" "$output_dir"
[ "$failures" -eq 0 ]
