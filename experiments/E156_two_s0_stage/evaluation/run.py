"""This stage authorizes S0 training only, without postprocessing."""
def main():
    raise PermissionError('E156 has no postprocessing or final evaluation authorization')
if __name__=='__main__':main()
