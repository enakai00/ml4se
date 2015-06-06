rm -f ml4se.zip
pushd ..
find ml4se/ | grep -E "(LICENSE|README|config_|scripts)" | zip ml4se/ml4se.zip -@
popd
