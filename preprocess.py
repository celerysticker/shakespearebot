import nltk

# Run this line the first time
# nltk.download('punkt')

def tokenize_lines(filename):
	all_tokens = []
	line_count = 14
	i = 0
	with open(filename) as f:
		for line in f.readlines():
			line = line.strip()
			try:
				int(line)
				i = 0 # reset line count
				continue
			except ValueError:
				pass

			if len(line) > 0:
				tokens = nltk.word_tokenize(line)
				all_tokens.append(tokens)
			i += 1
	return all_tokens

def print_tokens(all_tokens):
	print ""

tokens = tokenize_lines('data/shakespeare.txt')