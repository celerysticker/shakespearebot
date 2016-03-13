import nltk
import preprocess_words_only as pp
import preprocess as p
import random
import numpy as np

def tag_words(hmm):
	# tag each symbol individually
	sort_tag = {}
	for symbol in obs:
		max_state = np.argmax(hmm._outputs_vector(symbol))
		if max_state in sort_tag:
			sort_tag[max_state].append(symbol)
		else:
			sort_tag[max_state] = [symbol]

	for key in sort_tag:
		print "{0}\t{1}".format(key, sort_tag[key])

def print_A(hmm):
	# transition matrix
	print hmm._transitions_matrix()

def print_O(hmm):
	# emission matrix
	for symbol in obs:
		print hmm._outputs_vector(symbol)

def gen_random(hmm):
	# try generating sequence?
	rng = random.Random()
	rng.seed(0)
	for i in range(10):
		item = hmm.random_sample(rng, 20)
		print item

shakespeare = 'data/shakespeare_reduced.txt'
sequences = pp.tokenize_lines_words_only(shakespeare)
sequences_tag = p.format_tokens(sequences)

states = range(10)
obs = p.get_unique_obs(sequences)

trainer = nltk.tag.hmm.HiddenMarkovModelTrainer(states, obs)
hmm = trainer.train_unsupervised(sequences_tag, max_iterations=100)

# transition matrix
print_A(hmm)

# words in each state
tag_words(hmm)

# # not sure what this is...
# print hmm._outputs[1]._samples

