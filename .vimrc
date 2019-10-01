syntax on

let mapleader=","

nnoremap W $
nnoremap $ Q
nnoremap Q q
nnoremap q viwc
nnoremap B ^
nnoremap ^ B
noremap \ ,

filetype on
filetype indent on
filetype plugin on
filetype plugin indent on

set nocompatible
set spell
set hidden
set nocp
set nu
set autoread
set nobackup
set nowritebackup
set nowb
set noswapfile
set cursorline
set showcmd
set ruler
set showmatch
set ignorecase
" set smartcase
set infercase
set hlsearch
set incsearch
set autoindent
set smartindent
set expandtab
set wildmenu
set nrformats=
" set mouse=a
set history=100
set tabstop=4
set showtabline=2
set softtabstop=4
set shiftwidth=4
set cmdheight=2
set selection=exclusive
set selectmode=mouse,key
set foldmethod=marker
set updatetime=300
set shortmess+=c
set path+=/home/charlie/**

autocmd FileType ruby,eruby set omnifunc=rubycomplete#Complete
autocmd FileType python set omnifunc=pythoncomplete#Complete
autocmd FileType javascript set omnifunc=javascriptcomplete#CompleteJS
autocmd FileType html set omnifunc=htmlcomplete#CompleteTags
autocmd FileType css set omnifunc=csscomplete#CompleteCSS
autocmd FileType xml set omnifunc=xmlcomplete#CompleteTags
autocmd FileType java set omnifunc=javacomplete#Complet

""""""""""""""""""""""""""""""""""""""""

" plugin 'vim-plug' session begins:

call plug#begin('~/.vim/additional_plugins')

" solarized theme
Plug 'altercation/vim-colors-solarized'
" code autocomplete
Plug 'ycm-core/YouCompleteMe'
" file exploration tree
Plug 'vim-scripts/The-NERD-tree'
" macro code snippets autocomplete
Plug 'honza/vim-snippets'
Plug 'SirVer/ultisnips'
" git integration
Plug 'tpope/vim-fugitive'
" multi-language support
Plug 'sheerun/vim-polyglot'
" fast jump-tos
Plug 'easymotion/vim-easymotion'
" fast comments
Plug 'scrooloose/nerdcommenter'
" light status line theme
Plug 'itchyny/lightline.vim'
" bracket autocomplete and autodelete
Plug 'jiangmiao/auto-pairs'

call plug#end()

" plugin 'vim-plug' session ends.

""""""""""""""""""""""""""""""""""""""""

" plugin 'YouCompleteMe' session begins:

nnoremap <c-]> :YcmCompleter GoToDefinitionElseDeclaration<CR>

set completeopt=longest,menu
set completeopt-=preview
set signcolumn=yes

" let g:ycm_global_ycm_extra_conf='~/.vim/bundle/YouCompleteMe/cpp/ycm/.ycm_extra_conf.py'
" let g:ycm_key_invoke_completion='<M-;>'
" let g:ycm_use_ultisnips_completer=0
" let g:ycm_key_list_previous_completion=['<c-p>','<Up>']
" let g:ycm_key_list_select_completion=['<c-n>','<Down>']
let g:syntastic_ignore_files=[".*\.py$"] 
let g:ycm_complete_in_comments=1
let g:ycm_complete_in_strings=1
let g:ycm_collect_identifiers_from_comments_and_strings=1
let g:ycm_collect_identifiers_from_tag_files=1
let g:ycm_seed_identifiers_with_syntax=1
let g:ycm_confirm_extra_conf=0
let g:ycm_cache_omnifunc=0
let g:ycm_min_num_of_chars_for_completion=1
let g:ycm_error_symbol='>>'
let g:ycm_warning_symbol='>*'

" plugin 'YouCompleteMe' session ends.

""""""""""""""""""""""""""""""""""""""""

" plugin 'vim-colors-solarized' begins:

set background=dark

colorscheme solarized

" plugin 'vim-colors-solarized' ends.

""""""""""""""""""""""""""""""""""""""""

" plugin 'The-NERD-tree' session begins:

" autocmd vimenter * NERDTree

noremap <F3> :NERDTreeToggle<CR>

" let NERDTreeShowLineNumbers=1
" let NERDTreeAutoCenter=1
let NERDTreeWinSize=30
let NERDTreeShowBookmarks=1
let NERDTreeHighlightCursorline=1
let NERDTreeQuitOnOpen=1
let g:NERDTreeShowHidden=0
let NERDTreeIgnore=['\.pyc$','\.pyo$','\.py\$class$','\.obj$','\.o$','\.so$','\.egg$','^\.git$','__pycache__','\.DS_Store']

" plugin 'The-NERD-tree' session ends.

""""""""""""""""""""""""""""""""""""""""

" plugin 'ultisnips' session begins:

let g:UltiSnipsExpandTrigger="<tab>"
let g:UltiSnipsJumpForwardTrigger="<c-b>"
let g:UltiSnipsJumpBackwardTrigger="<c-z>"
let g:UltiSnipsEditSplit="vertical"

" plugin 'ultisnips' session ends.

""""""""""""""""""""""""""""""""""""""""

" plugin 'nerdcommenter' session begins:

let g:NERDSpaceDelims=1

" plugin 'nerdcommenter' session ends.

""""""""""""""""""""""""""""""""""""""""

" plugin 'lightline.vim' session begins:

set laststatus=2

" plugin 'lightline.vim' session ends.

""""""""""""""""""""""""""""""""""""""""

" plugin 'auto-pairs' session begins:

let &t_SI .= "\<Esc>[?2004h"
let &t_EI .= "\<Esc>[?2004l"

inoremap <special> <expr> <Esc>[200~ XTermPasteBegin()

function! XTermPasteBegin()
  set pastetoggle=<Esc>[201~
  set paste
  return ""
endfunction

" set pastetoggle=<F2>

" plugin 'auto-pairs' session ends.

""""""""""""""""""""""""""""""""""""""""
