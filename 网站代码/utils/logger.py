from datetime import datetime


def get_now_string():
    now = datetime.now()
    now = now.strftime("%Y%m%d%H%M%S")
    return string2date(now)


def string2date(string):
    return f"{string[:4]}-{string[4:6]}-{string[6:8]}"


if __name__ == '__main__':
    print(get_now_string())
