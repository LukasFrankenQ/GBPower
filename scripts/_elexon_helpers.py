from requests.exceptions import HTTPError
import time


def robust_request(getfunc, *args, max_retries=100, wait_time=5, **kwargs):

    for _ in range(max_retries):
        try:
            response = getfunc(*args, **kwargs)
            return response
        except HTTPError as http_err:
            print(f"HTTP error occurred: {http_err}")
        except Exception as err:
            print(f"Other error occurred: {err}")
        time.sleep(wait_time)

    raise Exception(f"Failed to get response after {max_retries} retries")