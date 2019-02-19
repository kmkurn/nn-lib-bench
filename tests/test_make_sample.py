from nnlibbench import make_sample


def test_make_sample():
    text = 'foo bar quux'
    sample = make_sample(text)
    assert sample['words'] == ['<s>', 'foo', 'bar', 'quux', '</s>']
    assert sample['chars'] == [['<s>'], list('foo'), list('bar'), list('quux'), ['</s>']]
