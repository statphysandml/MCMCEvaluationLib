from traceback import print_exc


def get_custom_measures_func():
    try:
        from custom_modules import compute_measures
        return compute_measures
    except ModuleNotFoundError:
        print("(Note: No module named custom_modules.py with a function compute_measures for potential custom measures found. "
              "You can compute custom measures by adding a corresponding .py file to your python path. In C++ "
              "this can be achieved by defining a python_modules_path via -DPYTHON_SCRIPTS_PATH=<relative or absolute "
              "path to custom_modules.py>.)")
        return None
    except ImportError:
        # print("No module named custom_modules.py with a function custom_load_data found."
        #       "You can compute custom measures by adding a corresponding .py file to your python path. In C++ "
        #       "this can be achieved by defining a python_modules_path via -DPYTHON_SCRIPTS_PATH=<relative or absolute"
        #       "path to custom_modules.py>.")
        return None
    except Exception as e:
        print_exc()


def get_custom_load_data_func():
    try:
        from custom_modules import custom_load_data
        return custom_load_data
    except ModuleNotFoundError:
        # print("No module named custom_modules.py with a function custom_load_data found."
        #       "You can compute custom measures by adding a corresponding .py file to your python path. In C++ "
        #       "this can be achieved by defining a python_modules_path via -DPYTHON_SCRIPTS_PATH=<relative or absolute"
        #       "path to custom_modules.py>.")
        return None
    except ImportError:
        # print("No module named custom_modules.py with a function custom_load_data found."
        #       "You can compute custom measures by adding a corresponding .py file to your python path. In C++ "
        #       "this can be achieved by defining a python_modules_path via -DPYTHON_SCRIPTS_PATH=<relative or absolute"
        #       "path to custom_modules.py>.")
        return None
    except Exception as e:
        print_exc()