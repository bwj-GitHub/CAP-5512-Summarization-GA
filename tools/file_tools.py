"""
Created on Aug 5, 2015

@author: brandon
"""

import os.path
import os



def fopen(filename, mode, **kwargs):
    try:
        try:
            os.makedirs(filename.rpartition("/")[0], exist_ok=False)
            print("Created Directories for " + str(filename))
        except OSError:
            # Directories already exist...
            pass
        return open(filename, mode, **kwargs)
    except:
        # open() for Python 2.7 doesn't have an encoding argument.
        return open(filename, mode)


def read_file(filename, preserve_whitespace=False, ignore_comments=False,
              ignore_markup=False):
    """Read filename and return a list containing each line in it."""

    lines = []
    with fopen(filename, "r", encoding="UTF-8") as fp:
        for line in fp:
            if ignore_markup and line[0] == "<":
                continue
            if ignore_comments and line[0] == "#":
                continue
            if preserve_whitespace is False:
                lines.append(line.strip())
            else:
                lines.append(line)
    return lines



def write_file(lines, filename):
    """Write a list of strings to filename."""
    
    with fopen(filename, "w", encoding="UTF-8") as fp:
        for line in lines:
            fp.write(line + "\n")
    print("Output written to: " + str(filename))


def read_file_as_string(filename, encoding="UTF-8"):
    string = ""
    with fopen(filename, "r", encoding=encoding, errors="ignore") as fp:
        for line in fp:
            string += line.strip() + " "
    return string.strip()

def read_file_hashed(filename):
    lines = []
    with fopen(filename, "r", encoding="UTF-8") as fp:
        for line in fp:
            line = line.strip()
            if line[0] != "#":
                lines.append(line)
    return lines


def read_document_groups(root_dir):
    lines = []
    # The directory name will be the tag
    dir_info = list(os.walk(root_dir))[0] # dir, sudirs, files
    print(dir_info)
    for subdir in dir_info[1]:
        print(subdir)
        dirname = root_dir + "/" + subdir + "/"
        for _, _, fs in os.walk(dirname):
            for fn in fs:
                fpath = dirname + fn
                lines.append((subdir, fn, read_file_as_string(fpath, encoding="UTF8")))
    return lines


def quick_walk_dirs(root_dir):
    # Get the names of all sub-directories in root_dir:
    dirs = list(os.walk(root_dir))[0][1]
    return dirs


def quick_walk(root_dir):
    # Get list of filenames in directory
    fnames = []
    files = list(os.walk(root_dir))[0][2] # dir, sudirs, files
    for file in files:
        fname = root_dir + "/" + file
        fnames.append(fname)
    return fnames
    


def mod_file_name(filename, new_dir=None, new_ext=None,
                  prepend="", postpend=""):
    parts = filename.rpartition("/")
    dir = parts[0] + "/"
    parts = parts[2].rpartition(".")
    base = parts[0]
    ext = parts[1] + parts[2]
    
    if new_dir:
        dir = new_dir
        if ext[-1] != "/":
            dir = dir + "/"
    if new_ext:
        ext = new_ext
        if ext[-1] != "/":
            ext = ext + "/"

    return dir + prepend + base + postpend + ext


# TESTING:
if __name__ == "__main__":
#     print("testing!")
#     lines = read_document_groups("./Input/3Train/")
#        
#     with fopen("20news-train.txt", "w", encoding="UTF8") as fp:
#         for line in lines:
#             l = line[0] + "\t" + line[2]
#             fp.write(l)
#             fp.write("\n")
#     print("Finished!")
    print( quick_walk("/Users/brandon/Documents/School/Spring_2016/CAP_6640") )
    print(quick_walk_dirs("/Users/brandon/Documents/School/Spring_2016/CAP_6640/Final_Project/Documents"))
