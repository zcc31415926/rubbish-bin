syntax on

set nocp

filetype on

filetype indent on

filetype plugin on
filetype plugin indent on
set completeopt=longest,menu

set nu

set history=100

set autoread

set nobackup
set nowb
set noswapfile

set mouse=a

set selection=exclusive
set selectmode=mouse,key

set cursorline

set showcmd

set paste

set ruler

set showmatch

set ignorecase

set hlsearch

set incsearch

set autoindent

set tabstop=4

set wildmenu
autocmd FileType ruby,eruby set omnifunc=rubycomplete#Complete
autocmd FileType python set omnifunc=pythoncomplete#Complete
autocmd FileType javascript set omnifunc=javascriptcomplete#CompleteJS
autocmd FileType html set omnifunc=htmlcomplete#CompleteTags
autocmd FileType css set omnifunc=csscomplete#CompleteCSS
autocmd FileType xml set omnifunc=xmlcomplete#CompleteTags
autocmd FileType java set omnifunc=javacomplete#Complet

call plug#begin('~/.vim/additional_plugins')

Plug 'liuchengxu/vim-which-key'
Plug 'skywind3000/quickmenu.vim'
Plug 'neoclide/coc.nvim'
Plug 'neoclide/coc-snippets'
Plug 'neomake/neomake'
Plug 'Yggdroot/LeaderF'
Plug 'sbdchd/neoformat'
Plug 'tpope/vim-fugitive'
Plug 'weirongxu/coc-explorer'
Plug 'sheerun/vim-polyglot'
Plug 'easymotion/vim-easymotion'
Plug 'scrooloose/nerdcommenter'
Plug 'neoclide/coc-pairs'
Plug 'liuchengxu/vista.vim'
Plug 'mg979/vim-visual-multi'
Plug 'Yggdroot/indentLine'
Plug 'itchyny/lightline.vim'

call plug#end()
