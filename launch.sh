export OMPI_MCA_pml=^ucx
export OMPI_MCA_btl=^ib
export OMPI_MCA_btl_tcp_if_include=nmn0
echo $LD_LIBRARY_PATH
python3 $1
