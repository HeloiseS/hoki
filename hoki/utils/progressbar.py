def print_progress_bar(iteration, total, prefix='', suffix='',
                       decimals=2, length=100, fill='█', printEnd="\r"):
    """
    Prints a progress bar

    Parameters
    ----------
    iteration: int
        Current iteration
    total: int
        Total number of iterations
    prefix: str, optional
        String before the bar. Default='' (blank)
    suffix: str, optional
        String after the bar. Default='' (blank)
    decimals: int, optional
        Decimal places quoted on the progress percentage
    length: int, optional
        Number of characters in the progress bar. Default==100 because 100 percent.
    fill: Unicode character, optional
        Character used to fill the progress bar. Default='█'
    printEnd:
        Last character when progress bar is finished. Default='\r'

    Returns
    -------
    None
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()

    # Thanks to https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
