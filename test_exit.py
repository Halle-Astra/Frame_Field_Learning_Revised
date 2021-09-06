def exit_test():
    import multiprocess
    print('process id : ', multiprocess.current_process().name)
    print('son process is running.')
    exit('son process is exiting.')
