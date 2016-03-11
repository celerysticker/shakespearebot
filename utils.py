import curses 
# from curses.ascii import isdigit
import nltk
from nltk.corpus import cmudict
import preprocess_words_only as pp

# Run this line the first time
# nltk.download('cmudict')
# if this doesn't work, try running nltk.download() in commandline

d = cmudict.dict()

# not sure why curses.ascii.isdigit doesn't work :(
def isdigit(d):
	try:
		int(d)
		return True
	except ValueError:
		return False

def nsyl(word):
	"""Calculates syllable count of single word"""
	lowercase = word.lower()
	if lowercase not in d:
		return -1
	else:
		return max([len([y for y in x if isdigit(y[-1])]) for x in d[lowercase]])

def get_syllables_str(line):
	"""Returns syllable count of string (multiple words)"""
	line = line.lower()
	words = pp.get_words_only_str(line)
	syllables = 0
	for word in words:
		syllables += nsyl(word)
	return syllables

# if __name__ == "__main__":
#	# for testing
# 	print nsyl("arithmetic")
# 	print get_syllables_str("This is: a poem.")