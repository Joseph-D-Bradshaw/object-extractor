#!/home/soulskrix/VirtualEnvs/BoundingBoxEnv/bin/python
import objectextractor
import resizeimages
import imagemerger
import randombackgroundselector

def display_options():
    print('1) Object Extractor')
    print('2) Resize Background Images (256x256)')
    print('3) Random Background Selector')
    print('4) Merge Images')
    print('5) Exit')

if __name__ == '__main__':
    print('-' * 30)
    print('Artificial Inflator Program.\nChoose a program to run..')
    print('-' * 30)
    display_options()

    option = 0
    while option < 1 or option > 5:
        option = int(input('> '))
        if option < 1 or option > 5 or not isinstance(option, int):
            print('Enter a valid option.')
            continue
        if option == 1:
            objectextractor.start_extracting()
        if option == 2:
            resizeimages.start_resizing()
        if option == 3:
            randombackgroundselector.move_half_to_merge()
        if option == 4:
            imagemerger.start_merging()
        if option == 5:
            exit()

        display_options()
        option = 0


