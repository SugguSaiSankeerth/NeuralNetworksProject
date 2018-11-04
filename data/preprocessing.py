"""
After extracting the RAR, we run this to move all the files into
the appropriate train/test folders.

Should only run this file once!
"""
import os
import os.path

def get_train_test_lists(version='01'):
    """
    Using one of the train/test files (01, 02, or 03), get the filename
    breakdowns we'll later use to move everything.
    """
    # Get our files based on version.
    test_file = os.path.join('TrainTest', 'testlist' + version + '.txt')
    train_file = os.path.join('TrainTest', 'trainlist' + version + '.txt')

    # Build the test list.
    with open(test_file) as fin:
        test_list = [row.strip() for row in list(fin)]

    # Build the train list. Extra step to remove the class index.
    with open(train_file) as fin:
        train_list = [row.strip() for row in list(fin)]
        train_list = [row.split(' ')[0] for row in train_list]

    # Set the groups in a dictionary.
    file_groups = {
        'train': train_list,
        'test': test_list
    }

    return file_groups

def move_files(file_groups):
    """This assumes all of our files are currently in _this_ directory.
    So move them to the appropriate spot. Only needs to happen once.
    """
    # Do each of our groups.
    directory = "UCF-101"
    for group, videos in file_groups.items():

        # Do each of our videos.
        for video in videos:

            # Get the parts.
            parts = video.split(os.path.sep)
            classname = parts[0]
            filename = parts[1]
            print(classname)
            print(filename)
            sourcefilename=os.path.join(directory,classname,filename)
            print(sourcefilename)

            # Check if this class exists.
            if not os.path.exists(os.path.join(group, classname)):
                print("Creating folder for %s/%s" % (group, classname))
                os.makedirs(os.path.join(group, classname))

            # Check if we have already moved this file, or at least that it
            # exists to move.
            if not os.path.exists(sourcefilename):
                print("Can't find %s to move. Skipping." % (sourcefilename))
                continue

            # Move it.
            dest = os.path.join(group, classname, filename)
            print("before Moving")
            print("Moving %s to %s" % (sourcefilename, dest))
            print("After Moving")
            os.system("cp "+sourcefilename+" "+dest)
            # os.rename(filename, dest)

    print("Done.")

def main():
    """
    Go through each of our train/test text files and move the videos
    to the right place.
    """
    # Get the videos in groups so we can move them.
    group_lists = get_train_test_lists()

    # Move the files.
    move_files(group_lists)

if __name__ == '__main__':
    main()
