# rustup environment for tcsh
if ( $?PATH ) then
    if ( "$PATH" !~ */Users/lakshaykumar/Documents/Playground/imc-prosperity-4/.cargo/bin* ) then
        setenv PATH "/Users/lakshaykumar/Documents/Playground/imc-prosperity-4/.cargo/bin:$PATH"
    endif
else
    setenv PATH "/Users/lakshaykumar/Documents/Playground/imc-prosperity-4/.cargo/bin"
endif
