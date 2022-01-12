from medai.metrics.report_generation.abn_match.matchers import (
    AnyGroupPattern,
    # AllGroupsPattern,
    AllWordsPattern,
)

UNC_PATTERN = AnyGroupPattern(
    # pylint: disable=line-too-long
    r'\b(stable|unchanged|can|maybe|may|could|would|might|possibly|possible|presumably|probable|suspicious|suspicion|suspect|suspected|suggestive|question|questionable|consistent|worrisome)\b',
    AllWordsPattern('similar', 'to', 'prior'),
    r'\b(versus|vs)\b',
)

### Pre-negation Uncertainty matches (i.e. greater precedence than negations)

# XXX is not excluded
# cannot exclude some XXX
# cannot rule out XXX
# no evidence to rule out XXX
# no new XXX
# no new area of XXX


### Post-negation Uncertainty matches (i.e. lower precedence than negation)


# can|maybe|may|could|would|might|possibly|possible|presumably|probable ... XXX
# suspicious|suspicion|suspected|suggestive XXXX
# questionable|consistent|worrisome ... XXX


# Stable|unchanged|similar to prior


# XXX be considered
# suspect|suspected|suspicious|suspicion
# consider|concern|favor

# XXX versus|vs YYY
# XXX or YYY

# May/might/would/could be XXX

# suspected XXX
# XXX suspected
# suggestive of XXX
# possibly reflecting a XXX

# maybe due to XXX
# maybe secondary to XXX
# can|could|may|would|possibly be due to XXX
# can|could|may|would|possibly related to XXX
# be|could|may|would be compatible with XXX
# may|could|would be XXX
# may|might|can|could be consistent/compatible with XXX

# may|could|would|might|possibly|can represent/reflect/indicate/include XXX
# may|could|would|might|possibly|can represent/reflect/indicate/include the presence of XXX

# question left XXX

# differential diagnosis includes XXX

# Correlation for symptoms of XXX
# correlate clinically for XXX
# correlate clinically for evidence|sign|signs|symptoms|symptom of XXX


# XXX could/might/may/possibly be present
# XXX could appear

# XXX is poorly|incompletely evaluated
# XXX is not well visualized/evaluated
# XXX was not appreciated
# XXX are not clearly seen

# obscuring the XXX
# obscured XXX
# XXX is obscured
