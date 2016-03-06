import nltk

# Run this line the first time
# nltk.download('punkt')

def tokenize_lines(filename):
	all_tokens = []
	line_count = 14
	i = 0
	with open(filename) as f:
		for line in f.readlines():
			if i == 0: # skip every 14th line
				i += 1
				continue

			if len(line) > 0:
				line = line.strip()
				tokens = nltk.word_tokenize(line)
				all_tokens.append(tokens)

			if i < line_count:
				i += 1
			else: # i == 14
				i = 0
	return all_tokens

print tokenize_lines('shakespeare.txt')