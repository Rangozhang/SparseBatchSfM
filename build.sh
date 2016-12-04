
if [ ! -f build ]; then
  echo "Creating a build directory"
  mkdir BUILD
fi

cd BUILD
cmake ..
make -j16
