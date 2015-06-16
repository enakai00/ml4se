rm -f ml4se.zip

pushd scripts
gzip -d *.gz
popd

pushd ..
find ml4se/ | grep -E "(LICENSE|README|config_|scripts)" | zip ml4se/ml4se.zip -@
popd

pushd scripts
gzip *.txt
popd
