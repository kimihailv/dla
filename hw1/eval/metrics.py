import editdistance


def calc_cer(src, target):
    if len(src) == 0 and len(target) == 0:
        return 1

    return editdistance.eval(target, src) / len(target)


def calc_wer(src, target):
    src_words = src.split()
    tgt_words = target.split()

    if len(src_words) == 0 and len(tgt_words) == 0:
        return 1

    return editdistance.eval(tgt_words, src_words) / len(tgt_words)
