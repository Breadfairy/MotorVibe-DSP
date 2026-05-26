################################################################################
# Imports                                                                      #
################################################################################

################################################################################
# variables/constants                                                          #
################################################################################
bpfoWarn = 1.20
bpfiWarn = 1.20

################################################################################
# main functions                                                               #
################################################################################


# Runs one small rules block on the current summed FFT band levels.
def runRules(sig):
    state = "watch"
    text = "band levels mixed"
    ratio = 0.0
    if sig["bpfiRms"] > 0.0:
        ratio = sig["bpfoRms"] / sig["bpfiRms"]
    if ratio >= bpfoWarn:
        state = "bpfo_watch"
        text = "bpfo band rising"
    if ratio > 0.0 and (1.0 / ratio) >= bpfiWarn:
        state = "bpfi_watch"
        text = "bpfi band rising"
    return {
        "state": state,
        "text": text,
        "bpfoRms": sig["bpfoRms"],
        "bpfiRms": sig["bpfiRms"],
        "ratio": ratio,
    }

