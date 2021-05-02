class HokiDialogue(object):

    def __init__(self):
        self.RED = '\u001b[31;1m'
        self.ORANGE = '\u001b[33;1m'
        self.GREEN = '\u001b[32;1m'
        self.BLUE = '\u001b[36;1m'
        self.ENDC = '\033[0m'
        self.BOLD = '\033[1m'
        self.UND = '\033[4m'
        self.BCKG='\u001b[46;1m'

    def info(self):
        return f'{self.GREEN}[---INFO---]{self.ENDC}'

    def running(self):
        return f'{self.ORANGE}[--RUNNING-]{self.ENDC}'

    def complete(self):
        return f'{self.BLUE}[-COMPLETE-]{self.ENDC}'

    def debugger(self):
        return f'{self.ORANGE}DEBUGGING ASSISTANT:{self.ENDC}'

    def error(self):
        return f'{self.RED}HOKI ERROR:{self.ENDC}'


dialogue = HokiDialogue()