def format_key(key):
    if type(key) == list:
        for k in key:
            assert type(k) == str, "keys should be strings"
    else:
        assert type(key) == str
    return key


class TransformByKey(object):
    """Performs data[outkey[0]], data[outkey[1]], ... = transform(data[key[0]], data[key[1]], ..)"""

    def __init__(self, transform, key=None, outkey=None, ignore_key_error=False, **kwargs):
        self.transform = transform
        if key is None:
            self.key = None
        else:
            self.key = format_key(key)
        if outkey is None:
            self.outkey = self.key
        else:
            self.outkey = format_key(outkey)
        self.ignore_key_error = ignore_key_error
        self.kwargs = kwargs

    def __call__(self, data):
        assert type(data) == dict, "Input data should be a dictionary, not a {}".format(type(data))
        try:
            if self.key is None:
                output = self.transform(**self.kwargs)
            elif type(self.key) == str:
                output = self.transform(data[self.key], **self.kwargs)
            else:
                inputs = [data[k] for k in self.key]
                import os
                # print(f"FILE:{os.path.split(__file__)[-1]}, var inspecting, transform:{self.transform},\n{dir(self.transform)}")
                # print(f"FILE:{os.path.split(__file__)[-1]},\ninputs:{inputs}\nkwargs:{self.kwargs}")
                output = self.transform(*inputs, **self.kwargs)
                # try:
                #     output = self.transform(*inputs, **self.kwargs)
                # except Exception as e :
                #     import multiprocess
                #     import os
                #     print(f"FILE:{os.path.split(__file__)[-1]}, process name:{multiprocess.current_process().name}, "
                #           f"this process encountered errors, var inspecting, \n"
                #           f"keys:{self.key}\n"
                #           )
                #     if 'gt_polygons' in self.key:
                #         print(f"FILE:{os.path.split(__file__)[-1]}, process name:{multiprocess.current_process().name}, "
                #           f"this process encountered errors, var inspecting, \n"
                #               f"gt_polygons in keys and value is {data['gt_polygons']}")
                #     print(f"FILE:{os.path.split(__file__)[-1]}, process name:{multiprocess.current_process().name}\t Exception is, ",e)
                # print(f"FILE:{os.path.split(__file__)[-1]}, inputs (shape:{inputs[0].shape}) have been transformed.")
                # print(f"FILE:{os.path.split(__file__)[-1]}, var inspecting, keys:{self.key}")

            if type(self.outkey) == str:
                data[self.outkey] = output
            else:
                assert type(output) == tuple, "Output should be tuple, not {} because outkey is {}".format(type(output), type(self.outkey))
                assert len(self.outkey) == len(output), "len(outkey) and len(output) should be the same for a 1-to-1 matching."
                for k, o in zip(self.outkey, output):
                    data[k] = o
        except KeyError as e:
            if not self.ignore_key_error:
                raise e
        return data