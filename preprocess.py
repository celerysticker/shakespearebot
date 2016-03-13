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

def format_tokens(list_tokens):
	"""Format tokens for nltk.hmm.tag"""
	all_tokens = []
	for sequence in list_tokens:
		token_seq = []
		for token in sequence:
			token_seq.append((token, ''))
		all_tokens.append(token_seq)
	return all_tokens

def get_unique_obs(list_tokens):
	all_tokens = []
	for sequence in list_tokens:
		for token in sequence:
			all_tokens.append(token)
	return list(set(all_tokens))


tokens = tokenize_lines('data/shakespeare.txt')
# print format_tokens(tokens)