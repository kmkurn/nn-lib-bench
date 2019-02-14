from nnlibbench import make_sample


def test_make_sample():
    text = 'foo bar quux'
    sample = make_sample(text)
    assert sample['words'] == ['<s>', 'foo', 'bar', 'quux']
    assert sample['chars'] == [['<s>'], list('foo'), list('bar'), list('quux')]
    assert sample['targets'] == ['foo', 'bar', 'quux', '</s>']
