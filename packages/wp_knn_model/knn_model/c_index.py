def c_index(true_labels, predictions):
    """ Calculates the concordance index (C-index) """

    true_labels = list(true_labels)
    predictions = list(predictions)

    n = 0
    h_sum = 0
    for i in range(len(true_labels)):
        t = true_labels[i]
        p = predictions[i]
        for j in range(i + 1, len(true_labels)):
            nt = true_labels[j]
            np = predictions[j]
            if t != nt:
                n += 1
                if (p < np and t < nt) or (p > np and t > nt):
                    h_sum += 1
                elif p == np:
                    h_sum += 0.5
    # To avoid 'ZeroDivisionError' exception
    if n == 0:
        return h_sum
    return h_sum / n
