echo "project( my_run_picar )" > CMakeLists.txt
echo "find_package( OpenCV REQUIRED )" >> CMakeLists.txt
echo "include_directories( \"../\" )" >> CMakeLists.txt
echo "add_executable( my_run_picar $1 )" >> CMakeLists.txt
echo "target_link_libraries( my_run_picar \${OpenCV_LIBS} )" >> CMakeLists.txt

cmake .
make -j2
./my_run_picar