import numpy as np
from collections import Counter


def load_data(in_file, max_example=None, relabeling=True):
    """
        load CNN / Daily Mail data from {train | dev | test}.txt
        relabeling: relabel the entities by their first occurence if it is True.
    """

    documents = []
    questions = []
    answers = []
    num_examples = 0
    with open(in_file, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            question = line.strip().lower()
            answer = f.readline().strip()
            document = f.readline().strip().lower()

            if relabeling:
                q_words = question.split(' ')
                d_words = document.split(' ')
                assert answer in d_words

                entity_dict = {}
                entity_id = 0
                for word in d_words + q_words:
                    if (word.startswith('@entity')) and (word not in entity_dict):
                        entity_dict[word] = '@entity' + str(entity_id)
                        entity_id += 1

                q_words = [entity_dict[w] if w in entity_dict else w for w in q_words]
                d_words = [entity_dict[w] if w in entity_dict else w for w in d_words]
                answer = entity_dict[answer]

                question = ' '.join(q_words)
                document = ' '.join(d_words)

            questions.append(question)
            answers.append(answer)
            documents.append(document)
            num_examples += 1

            f.readline()
            if (max_example is not None) and (num_examples >= max_example):
                break
                
    print('#Examples: %d' % len(documents))
    return (documents, questions, answers)


def build_dict(sentences, max_words=50000):
    """
        Build a dictionary for the words in `sentences`.
        Only the max_words ones are kept and the remaining will be mapped to <UNK>.
    """
    word_count = Counter()
    for sent in sentences:
        for w in sent.split(' '):
            word_count[w] += 1

    ls = word_count.most_common(max_words)
    print('#Words: %d -> %d' % (len(word_count), len(ls)))
    for key in ls[:5]:
        print(key)
    print('...')
    for key in ls[-5:]:
        print(key)

    # leave 0 to UNK
    # leave 1 to delimiter |||
    return {w[0]: index + 2 for (index, w) in enumerate(ls)}


def gen_embeddings(word_dict, dim, in_file=None):
    """
        Generate an initial embedding matrix for `word_dict`.
        If an embedding file is not given or a word is not in the embedding file,
        a randomly initialized vector will be used.
    """

    num_words = len(word_dict) + 2
    embeddings = np.random.uniform(size=(num_words, dim))
    print('Embeddings: %d x %d' % (num_words, dim))

    if in_file is not None:
        print('Loading embedding file: %s' % in_file)
        pre_trained = 0
        for line in open(in_file).readlines():
            sp = line.split()
            assert len(sp) == dim + 1 # word + embeddings ..
            if sp[0] in word_dict:
                pre_trained += 1
                embeddings[word_dict[sp[0]]] = [float(x) for x in sp[1:]]
        print('Pre-trained: %d (%.2f%%)' %
              (pre_trained, pre_trained * 100.0 / num_words))
    return embeddings
