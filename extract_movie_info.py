summaries = open("MovieSummaries/plot_summaries.txt", "r")
metadata = open("MovieSummaries/movie.metadata.tsv", "r")

movies = dict() # An entry, keyed by wikipedia ID, is (revenue, summary, nlp_summary)

# Iterate over metadata, extracting wikipedia ID and revenue
for line in metadata:
    ID = int(line.split("\t")[0])
    revenue = line.split("\t")[4]
    if revenue:
        movies[ID] = [int(revenue)]

# Iterate over summaries, extracting wikipedia ID and summary
for line in summaries:
    ID = int(line.split("\t")[0])
    if ID in movies:
        movies[ID].append(line.split("\t")[1])

# Remove movies that have a revenue but no summary
for key, values in movies.items():
    if len(values) < 2:
        movies.pop(key)

# We now have 7588 entries movies, keyed by the Wikipedia ID and containing (revenue, summary) lists

# Feature Extraction stuff here

# Prediction Stuff here

# Testing stuff here
