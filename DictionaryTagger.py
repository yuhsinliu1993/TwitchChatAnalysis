import yaml, os

class DictionaryTagger(object):

	def __init__(self, yaml_dir):
		files = [open(os.path.join(yaml_dir, fn), 'r') for fn in os.listdir(yaml_dir)]
		dictionaries = [yaml.load(dict_file) for dict_file in files]
		map(lambda x: x.close(), files)
		
		self.dictionary = {}
		self.max_key_size = 0
		
		for curr_dict in dictionaries:
			for key in curr_dict:
				if key:
					if key in self.dictionary:
						self.dictionary[key].extend(curr_dict[key])
					else:
						self.dictionary[key] = curr_dict[key]
						self.max_key_size = max(self.max_key_size, len(key))

	def tag(self, token_lists):
		tagged = []
		for tokens in token_lists:
			t = [self.tag_sentence(tokens[0])]
			t.extend(tokens[1:])
			tagged.append(t)
		return tagged

	def tag_sentence(self, tokens, tag_with_lemmas=False):
		"""
		the result is only one tagging of all the possible ones.
		The resulting tagging is determined by these two priority rules:
			- "longest matches" have higher priority
			- search is made from left to right
		"""
		tag_sentence = []
		N = len(tokens)
		if self.max_key_size == 0:
			self.max_key_size = N
		i = 0

		while (i < N):
			j = min(i + self.max_key_size, N) # avoid overflow
			tagged = False
			while (j > i):
				expression_form = ' '.join([token[0] for token in tokens[i:j]]).lower()
				expression_lemma = ' '.join([token[1] for token in tokens[i:j]]).lower()
				prop = ' '.join([token[3] for token in tokens[i:j]]) # properties
				if tag_with_lemmas:
					literal = expression_lemma
				else:
					literal = expression_form
				if literal in self.dictionary:
					# self.logger.debug("found: %s" % literal)
					is_single_token = j - i == 1
					original_position = i
					i = j
					taggings = [tag for tag in self.dictionary[literal]] # positive, negative
					tagged_expression = (expression_form, expression_lemma, taggings, prop)
					if is_single_token: # if the tagged literal is a single token, conserve its previous taggings:
						original_token_tagging = tokens[original_position][2]
						tagged_expression[2].extend(original_token_tagging) # add original pos tag
					tag_sentence.append(tagged_expression)
					tagged = True
				else:
					j = j - 1

			if not tagged:
				tag_sentence.append(tokens[i])
				i += 1
		
		return tag_sentence
