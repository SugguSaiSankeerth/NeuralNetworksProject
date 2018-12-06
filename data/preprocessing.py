
import os
import os.path

def get_train_test_lists(version='01'):

    test_file = os.path.join('TrainTest', 'testlist' + version + '.txt')
    train_file = os.path.join('TrainTest', 'trainlist' + version + '.txt')


    with open(test_file) as fin:
        test_list = [row.strip() for row in list(fin)]

    
    with open(train_file) as fin:
        train_list = [row.strip() for row in list(fin)]
        train_list = [row.split(' ')[0] for row in train_list]


    file_groups = {
        'train': train_list,
        'test': test_list
    }

    return file_groups

def move_files(file_groups):
 
    directory = "UCF-101"
    for group, videos in file_groups.items():

     
        for video in videos:

            parts = video.split(os.path.sep)
            classname = parts[0]
            filename = parts[1]
            print(classname)
            print(filename)
            sourcefilename=os.path.join(directory,classname,filename)
            print(sourcefilename)

            if not os.path.exists(os.path.join(group, classname)):
                print("Creating folder for %s/%s" % (group, classname))
                os.makedirs(os.path.join(group, classname))

           
            if not os.path.exists(sourcefilename):
                print("Can't find %s to move. Skipping." % (sourcefilename))
                continue

          
            dest = os.path.join(group, classname, filename)
         
            os.system("cp "+sourcefilename+" "+dest)
        

    print("Done.")

def main():

    group_lists = get_train_test_lists()


    move_files(group_lists)

if __name__ == '__main__':
    main()
