class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def printHeader(header):
    print(bcolors.HEADER + bcolors.BOLD + bcolors.UNDERLINE + header + bcolors.ENDC)

def printText(text):
    print(text)

def printWarning(warning):
    print(bcolors.WARNING + warning  + bcolors.ENDC)

def printError(error):
    print(bcolors.FAIL + error  + bcolors.ENDC)

def printBlue(text):
    print(bcolors.OKBLUE + text + bcolors.ENDC)

def printGreen(text):
    print(bcolors.OKGREEN + text + bcolors.ENDC)