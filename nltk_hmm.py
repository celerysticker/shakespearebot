import nltk
import preprocess_words_only as pp
import preprocess as p
import random

shakespeare = 'data/shakespeare_reduced.txt'
sequences = pp.tokenize_lines_words_only(shakespeare)
sequences_tag = p.format_tokens(sequences)
# sequences = [[('test', '')], [('test', '')]]
states = range(5)
obs = p.get_unique_obs(sequences)

trainer = nltk.tag.hmm.HiddenMarkovModelTrainer(states, obs)
hmm = trainer.train_unsupervised(sequences_tag, max_iterations=100)

# # not sure what this is...
# print hmm._outputs[1]._samples

# try generating sequence?
rng = random.Random()
rng.seed(0)
for i in range(10):
	item = hmm.random_sample(rng, 20)
	print item