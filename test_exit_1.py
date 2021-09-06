from test_exit import exit_test

def main():
    import multiprocess
    print('process id of main,',multiprocess.current_process().name)
    print('main process is running.')
    exit_test()
    exit('main process is exiting.')
if __name__ == '__main__':
    main()
