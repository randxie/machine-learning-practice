
class Trie(object):
    """
    https://en.wikipedia.org/wiki/Trie
    """
    def __init__(self):
        self.trie = {}

    def insert(self, word):
        if len(word) > 0:
            ref = self.trie
            for w in word:
                if w not in ref:
                    ref[w] = {}
                ref = ref[w]

            ref['stop'] = 1

    def search(self, word):
        ref = self.trie
        for w in word:
            if w not in ref:
                return False
            else:
                ref = ref[w]

        if 'stop' in ref:
            return True
        else:
            return False

    def starts_with(self, prefix):
        ref = self.trie
        for w in prefix:
            try:
                ref = ref[w]
            except Exception as e:
                print(e)
                return False

        return True


if __name__ == '__main__':
    data = ['a', 'b', 'c', 'ab', 'abc']
    mdl = Trie()
    for w in data:
        mdl.insert(w)

    print(mdl.search('abd'))
    print(mdl.search('abc'))
    print(mdl.starts_with('d'))
    print(mdl.starts_with('a'))

