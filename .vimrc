" XXX: AVOID USING `autocmd` !!!
" :verbose set X to check detailed settings of variable X
" :help Y to check details of attribute Y

" ----------------
" vim-plug plugins
" ----------------
call plug#begin('~/.vim/additional_plugins')
">>>>>>>>>> ui themes <<<<<<<<<<
" molokai color scheme
Plug 'tomasr/molokai'
" airline status line
Plug 'vim-airline/vim-airline'
Plug 'vim-airline/vim-airline-themes'
" background dimming color scheme
Plug 'junegunn/limelight.vim',{'on':'Limelight'}
" multi-color bracket color scheme
Plug 'frazrepo/vim-rainbow'
" vertical line indent indicator
Plug 'Yggdroot/indentLine'
">>>>>>>>>> code auto-completers <<<<<<<<<<
" youcompleteme code auto-completer
Plug 'ycm-core/YouCompleteMe'
" snippet auto-completer
Plug 'SirVer/ultisnips'
" personal snippet repo
Plug 'honza/vim-snippets'
" github copilot online code completer
Plug 'github/copilot.vim'
">>>>>>>>>> vim checkers & linters & debuggers <<<<<<<<<<
" ale syntax checker
Plug 'dense-analysis/ale'
" in-vim debugger
Plug 'Shougo/vimproc.vim'
Plug 'idanarye/vim-vebugger'
">>>>>>>>>> editing helpers <<<<<<<<<<
" fast jump-to module
Plug 'easymotion/vim-easymotion'
" fast commentator
Plug 'scrooloose/nerdcommenter'
" auto bracket pairing module
Plug 'jiangmiao/auto-pairs'
" fast bracket adder
Plug 'tpope/vim-surround'
" copy & paste menu
Plug 'vim-scripts/YankRing.vim'
" undo visualizer
Plug 'mbbill/undotree'
" plugin command repetition recorder with `.`
Plug 'tpope/vim-repeat'
">>>>>>>>>> file explorer <<<<<<<<<<
" filetree explorer engine
Plug 'preservim/nerdtree',{'on':'NERDTreeToggleVCS'}
" git indicators for nerdtree
Plug 'Xuyuanp/nerdtree-git-plugin',{'on':'NERDTreeToggleVCS'}
" filetype-specific highlighting for nerdtree
Plug 'tiagofumo/vim-nerdtree-syntax-highlight',{'on':'NERDTreeToggleVCS'}
" state preservation and restoration for nerdtree
Plug 'scrooloose/nerdtree-project-plugin',{'on':'NERDTreeToggleVCS'}
" current file highlighting and in-panel file closing for nerdtree
Plug 'PhilRunninger/nerdtree-buffer-ops',{'on':'NERDTreeToggleVCS'}
" visual mode for nerdtree
Plug 'PhilRunninger/nerdtree-visual-selection',{'on':'NERDTreeToggleVCS'}
">>>>>>>>>> extra functionalities <<<<<<<<<<
" global pattern / file fuzzy search engine
Plug 'Yggdroot/LeaderF',{'do':':LeaderfInstallCExtension','on':['Leaderf','LeaderfFile']}
" embedded terminal
Plug 'skywind3000/vim-terminal-help'
" semantic tag module
Plug 'liuchengxu/vista.vim'
" markdown preview visualizer
Plug 'suan/vim-instant-markdown'
" basic latex module
Plug 'lervag/vimtex',{'for':'tex'}
" latex preview visualizer
Plug 'xuhdev/vim-latex-live-preview',{'for':'tex'}
call plug#end()

" -------------------------------
" built-in variables & attributes
" -------------------------------
" language & encoding
language en_US.UTF-8
set encoding=utf8
" vi incompatibility
set nocompatible
" allowing background unsaved buffer
set hidden
" relative line number display with the current line number absolute
set number
set relativenumber
" file auto-reloading on external changes with checking interval 1000ms
set autoread
set updatetime=1000
" no backup / temp files
set nobackup
set nowritebackup
set noswapfile
" undo operation preservation with maximum 1000 history
set undofile
set history=1000
set undodir=~/.vim/.undo//
" current line highlighting
set cursorline
" vim command display on bottom-right
set showcmd
" cursor location display
set ruler
" matching bracket display
set showmatch
" case-insensitive auto-completion
set ignorecase
set infercase
set wildignorecase
" first-occurrence matching on every character input
set incsearch
" smart auto-indent with 4 <Space>s
set autoindent
set smartindent
set shiftwidth=4
" <Tab> to 4 <Space>s conversion in display
set tabstop=4
set expandtab
" trailing space highlighting
set list
set listchars=trail:\\u2588
" list:longest & full command line auto-completion
" first <Tab>: max common substring auto-completion with a candidate list
" second <Tab>: recursive selection
set wildmenu
set wildmode=list:longest,full
" forcing decimal in quick addition (<C-a>) / subtraction (<C-x>)
" conflicting with current settings
set nrformats=
" enabling mouse in normal and visual modes
set mouse=nv
" sign column for error and warning indicators
set signcolumn=yes
" a red column code length indicator
set colorcolumn=74
" line wrapping only on word separators
set linebreak
" multi-tabpage information line display
set showtabline=2
" command line height
set cmdheight=1
" disabling select mode
set selectmode=
" total occurrence count & current occurrence index display
set shortmess-=S
" code folding according to the indent level (off on startup)
set foldlevel=99
set foldmethod=indent
" auto-jumping into the root folder of the current file
set autochdir
" auto-complete menu display even if there is only one candidate
set completeopt=menu,menuone
" concealed text replaced with <Space>s
set conceallevel=1
" <Leader> key
let mapleader=','
" disabling matched bracket highlighting
let g:loaded_matchparen=1
" latex as the default tex filetype
let g:tex_flavor='latex'
" concealment for:
" a: accents & ligatures
" b: bolds & italics
" d: delimiters
" g: greek letters
" m: math symbols
" s: superscripts & subscripts
let g:tex_conceal='abdgms'

" ----------------
" built-in keymaps
" ----------------
" exiting insert / visual modes
inoremap jk <Esc>
" fast tabpage switching
nnoremap <Tab> gt
" under-the-cursor file opening
nnoremap gf <C-w>gf
" fast jump-out-of-brackets
inoremap <C-Space> <Esc>la
" visual-line cursor moving & centering
nnoremap j gjzz
nnoremap k gkzz
" file opening in a new tabpage
nnoremap m :tabe<Space>
" file saving
nnoremap t :w<CR>
" file closing
nnoremap Q :q<CR>
" jumping to the start / end of lines
nnoremap W $
nnoremap B ^
" word selection & deletion & entering insert mode
nnoremap q viwc
" indent & outdent
nnoremap < <<
nnoremap > >>
" redo
nnoremap U <C-r>
" block folding
nnoremap <Space> za
" prefix for switching between panels
nnoremap <C-a> <C-w>
" occurrence replacing
" replacing `foo` with `bar`: :%s/foo/bar/gc
" g: global, c: confirmation with:
" y (replace and continue), n (skip and continue)
" a (replace all), q / <Esc> (quit) / l (replace and quit)
" <C-e> (scroll up) / <C-y> (scroll down)
nnoremap <C-h> :%s/

" ------------------
" built-in omnifuncs
" ------------------
" python auto-completion done by youcompleteme
"autocmd FileType python set omnifunc=pythoncomplete#Complete
autocmd FileType ruby,eruby set omnifunc=rubycomplete#Complete
autocmd FileType javascript set omnifunc=javascriptcomplete#CompleteJS
autocmd FileType html set omnifunc=htmlcomplete#CompleteTags
autocmd FileType css set omnifunc=csscomplete#CompleteCSS
autocmd FileType xml set omnifunc=xmlcomplete#CompleteTags
autocmd FileType java set omnifunc=javacomplete#Complet

" -------
" molokai
" -------
" color scheme
colorscheme desert

" ----------------------------------
" vim-airline
" ----------------------------------
" | A | B | C        X | Y | Z | W |
" ----------------------------------
" status line color scheme
let g:airline_theme='molokai'
" vista integrated into status line
let g:airline#extensions#vista#enabled=1
" ale integrated into status line
let g:airline#extensions#ale#enabled=1
" section c:
" F: absolute filepath
" m: modified flag
" r: readonly flag
" w: preview window flag
let g:airline_section_c='%F%m%r%w'
" section z:
" p: cursor relative location
" l: line index
" c: column index
let g:airline_section_z='%p%% L%l C%c'
" section w: blank
let g:airline_section_warning=''

" -------------
" limelight.vim
" -------------
" focus mode with dimmed context
nnoremap <Leader>d :Limelight!! 0.7<CR>

" -----------
" vim-rainbow
" -----------
" multi-color brackets enabled
let g:rainbow_active=1

" ----------
" indentLine
" ----------
" no conceal mode in indentline
let g:indentLine_setConceal=0

" -------------
" YouCompleteMe
" -------------
" fast-jump to definition / declaration
nnoremap <c-]> :YcmCompleter GoToDefinitionElseDeclaration<CR>
" no floating window in completion
let g:ycm_auto_hover=''
" floating window for function / variable details
nnoremap <F4> <Plug>(YCMHover)
" single arbitrary character to trigger semantic auto-complete
let g:ycm_semantic_triggers={'python':['re!\w{1}']}
" ycm-driven semantic highlighting
let g:ycm_enable_semantic_highlighting=1
" in-line argument name display
let g:ycm_enable_inlay_hints=1
" no argument name display in insert mode
let g:ycm_clear_inlay_hints_in_insert_mode=1
" auto-completion in comments
let g:ycm_complete_in_comments=1
" auto-completion in strings
let g:ycm_complete_in_strings=1
" collecting identifiers from comments and strings
let g:ycm_collect_identifiers_from_comments_and_strings=1
" collecting identifiers from tag files
let g:ycm_collect_identifiers_from_tag_files=1
" external syntax database used for identifiers
let g:ycm_seed_identifiers_with_syntax=1
" no confirmation when using custom config file
let g:ycm_confirm_extra_conf=0
" compatibility with all omnifuncs with speed sacrifice
let g:ycm_cache_omnifunc=0
" minimum 1 characters to trigger auto-completion
let g:ycm_min_num_of_chars_for_completion=1

" ---------
" ultisnips
" ---------
" snippet auto-complete trigger
let g:UltiSnipsExpandTrigger='<c-f>'
" switching between to-be-filled parameter blanks
let g:UltiSnipsJumpForwardTrigger='<c-f>'
let g:UltiSnipsJumpBackwardTrigger='<c-b>'
let g:UltiSnipsEditSplit='vertical'
" single quotes for python snippets
let g:ultisnips_python_quoting_style='single'
let g:ultisnips_python_triple_quoting_style='single'

" ---
" ale
" ---
" activated fixers for all files
let g:ale_fixers=['prettier','eslint','trim_whitespace']
" activated fixers for python files
let g:ale_fixers={'python':['autoflake','autopep8','autoimport','pycln']}
" error and warning display on all problematic lines
let g:ale_virtualtext_cursor='all'
" signs for errors and warnings
let g:ale_sign_error='X'
let g:ale_sign_warning='!'
" module toggle
nnoremap <F6> :ALEToggle<CR>
" code auto-fixing
nnoremap <F7> :ALEFix<CR>
" fast jump to previous problematic line
nnoremap <C-k> :ALEPreviousWrap<CR>
" fast jump to next problematic line
nnoremap <C-j> :ALENextWrap<CR>

" ------------
" vim-vebugger
" ------------
" signals for breakpoints and current line in sign column
let g:vebugger_breakpoint_text='>'
let g:vebugger_currentline_text='@'
" in-vim PDB trigger
nnoremap <F5> :VBGstartPDB<Space>
">>>>>>>>>> hard-coded vebugger leader key <<<<<<<<<<
" breakpoint toggle at the current line
nnoremap `b :VBGtoggleBreakpointThisLine<CR>
" continue running to the next breakpoint
nnoremap `c :VBGcontinue<CR>
" stepping over the current line
nnoremap `<Space> :VBGstepOver<CR>
" stepping in the detailed implementation of the current line
nnoremap `i :VBGstepIn<CR>
" stepping out of the current detailed implementation
nnoremap `o :VBGstepOut<CR>
" querying a variable
nnoremap `e :VBGeval<Space>
" exiting PDB
nnoremap `q :VBGkill<Space>

" --------------
" vim-easymotion
" --------------
" bi-directional fast jump-to
nnoremap f <Plug>(easymotion-s)

" -------------
" nerdcommenter
" -------------
" no <Space> added between the comment characters and the code
let g:NERDSpaceDelims=0
" compact multi-line comment style
let g:NERDCompactSexyComs=1
" trailing whitespace trimming when uncommenting a line
let g:NERDTrimTrailingWhitespace=1
" no default keymap loaded
let g:NERDCreateDefaultMappings=0
" commentation toggle in normal and visual modes
nnoremap <Leader>c <Plug>NERDCommenterToggle
vnoremap <Leader>c <Plug>NERDCommenterToggle

" ----------
" auto-pairs
" ----------
" function for auto-bracket-pairing
let &t_SI .= "\<Esc>[?2004h"
let &t_EI .= "\<Esc>[?2004l"
inoremap <special> <expr> <Esc>[200~ XTermPasteBegin()
function! XTermPasteBegin()
    set pastetoggle=<Esc>201~
    set paste
    return ''
endfunction

" ------------
" vim-surround
" ------------
" key + right bracket for surrounding bracket addition
nmap ( ysiw
" key + right bracket for surrounding bracket removal
nmap ) ds

" ------------
" YankRing.vim
" ------------
" maximum 50 items in history
let g:yankring_max_history=50
" history recording path
let g:yankring_history_dir='~/.vim/.yankring'
" yankring panel toggle
nnoremap <Leader>p :YRShow<CR>

" --------
" undotree
" --------
" panel toggle
nnoremap <Leader>u :UndotreeToggle<CR>

" -------
" LeaderF
" -------
" fuzzy file searching
nnoremap <Leader>o :LeaderfFile<CR>
" fuzzy pattern matching
nnoremap <Leader>f :Leaderf rg<CR>

" -----------------
" vim-terminal-help
" -----------------
" in-vim terminal panel toggle
let g:terminal_key='<C-n>'

" ---------
" vista.vim
" ---------
" taglist panel toggle
nnoremap <F2> :Vista!!<CR>

" --------
" nerdtree
" --------
" panel toggle
noremap <F3> :NERDTreeToggleVCS<CR>
" panel size
let NERDTreeWinSize=40
" bookmark display in the panel
let NERDTreeShowBookmarks=1
" file panel closed after opening a file
" bookmark panel closed after opening a bookmark
let NERDTreeQuitOnOpen=3
" files omitted from the panel
let NERDTreeIgnore=['\.pyc$','\.pyo$','\.py\$class$','\.obj$','\.o$','\.so$','\.egg$','^\.git$','__pycache__','\.DS_Store']

" --------------------
" vim-instant-markdown
" --------------------
" re-rendering only after afk / leaving insert mode / saving
let g:instant_markdown_slow=1
" allowing script running behavior
let g:instant_markdown_allow_unsafe_content=1
" tex-style code rendering
let g:instant_markdown_mathjax=1
" port for rendering
let g:instant_markdown_port=8888

" ----------------------
" vim-latex-live-preview
" ----------------------
" pdf viewer for rendering
let g:livepreview_previewer='gv'
" re-rendering trigger
nnoremap <F12> :LLPStartPreview<CR>

