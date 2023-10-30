__author__ = 'tylin'


from .bleu import Bleu
from .meteor import Meteor
from .rouge import Rouge
from .cider import Cider
from .spice.spice import Spice
from .tokenizer.ptbtokenizer2 import PTBTokenizer

def compute_scores(gts, gen,spice=False):
    if(spice):
        metrics = (Bleu(), Meteor(), Rouge(), Cider(),Spice())
    else:
        metrics = (Bleu(), Meteor(), Rouge(), Cider())
    all_score = {}
    all_scores = {}
    for metric in metrics:
        score, scores = metric.compute_score(gts, gen)
        all_score[str(metric)] = score
        all_scores[str(metric)] = scores

    return all_score, all_scores

def compute_scores2(gts, gen):
    metrics = (Bleu(), Meteor(), Rouge(), Cider(),Spice())
    all_score = {}
    all_scores = {}
    for metric in metrics:
        score, scores = metric.compute_score(gts, gen)
        all_score[str(metric)] = score
        all_scores[str(metric)] = scores

    return all_score, all_scores

def compute_bleus(gts, gen):
    metrics = (Bleu(),)
    all_score = {}
    all_scores = {}
    for metric in metrics:
        score, scores = metric.compute_score(gts, gen)
        all_score[str(metric)] = score
        all_scores[str(metric)] = scores

    return all_score, all_scores


def compute_ciders(gts, gen):
    metrics = (Cider(),)
    # all_score = {}
    # all_scores = {}
    for metric in metrics:
        score, scores = metric.compute_score(gts, gen)
        # all_score[str(metric)] = score
        # all_scores[str(metric)] = scores

    return score, scores