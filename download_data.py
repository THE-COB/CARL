import wget
import tyro
import shutil

def main(obj: str):
    url = f"https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/{obj}.obj"
    filename = wget.download(url)
    shutil.move(filename, f"objs/{filename}") 

tyro.cli(main)

