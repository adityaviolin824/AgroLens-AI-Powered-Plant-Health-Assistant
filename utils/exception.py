import sys   
from utils.logger import logging
import traceback

def error_message_details(error, error_detail: sys = None):
    if error_detail is None:
        error_detail = sys
    """
    Function to extract detailed error information using sys.exc_info().
    
    Parameters:
        error: The actual error/exception object.
        error_detail (sys): sys module reference, used to fetch exception details.
    
    Why do we do this?
        - When an exception occurs, Python stores information about it.
        - sys.exc_info() returns a tuple: (exception type, exception value, traceback object).
        - The traceback object helps us trace where exactly in the code the error happened.
        - This is very useful for debugging and logging.
    """


    _, _, exc_tb = error_detail.exc_info()

    file_name = "<unknown>"
    line_no = -1

    tb = getattr(error, "__traceback__", None)
    if tb is not None:
        frames = traceback.extract_tb(tb)
        if frames:
            last = frames[-1]
            file_name = last.filename
            line_no = last.lineno
    elif exc_tb is not None:

        try:
            file_name = exc_tb.tb_frame.f_code.co_filename 
            line_no = exc_tb.tb_lineno
        except Exception:
            pass



    error_message="Error occured in python script name [{0}] line number [{1}] error message[{2}]".format(
    file_name,line_no,str(error)) 

    return error_message

    

class CustomException(Exception): 

    def __init__(self,error_message,error_detail:sys):
        super().__init__(str(error_message))
        self.error_message=error_message_details(error_message,error_detail=error_detail)
    
    def __str__(self): 
        return self.error_message
    


