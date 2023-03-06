% ccr.cls
% Template for CCR Articles (very WIP)
% 2023 Wouter van Atteveldt

\ProvidesClass{ccr}[2023-02-03 v0.01]
\NeedsTeXFormat{LaTeX2e}

\LoadClass[twoside]{article}

%%%%%%%%%%%%% OVRERALL PAGE LAYOUT %%%%%%%%%%%%%%%%%%%
\RequirePackage[papersize={6.53in,9.61in},
  left=1.1in,right=1.1in,top=1in,textheight=7.25in]{geometry}

\usepackage[proportional,scaled=1]{erewhon}
\usepackage[portuguese]{babel}
\RequirePackage[protrusion=true,expansion=true,final,babel]{microtype}

\linespread{1.1}

%%%%%%%%%%%%%% Information from authors %%%%%%%%%%%%%%%%%%%%%
\RequirePackage{ifthen}

\def\@shorttitle{}
\newcommand{\shorttitle}[1]{\def\@shorttitle{#1}}
\newcommand{\show@shorttitle}{%
  \ifthenelse{\equal{\@shorttitle}{}}%
  {\MakeLowercase{\@title}}{\MakeLowercase{\@shorttitle}}%
}

\def\@shortauthors{(please specify \shortauthors)}
\newcommand{\shortauthors}[1]{\def\@shortauthors{#1}}
\newcommand{\show@shortauthors}{\@shortauthors}

\def\@volume{X}
\newcommand{\volume}[1]{\def\@volume{#1}}
\newcommand{\show@volume}{\@volume}
\def\@pubnumber{Y}
\newcommand{\pubnumber}[1]{\def\@pubnumber{#1}}
\newcommand{\show@pubnumber}{\@pubnumber}
\def\@pubyear{20xx}
\newcommand{\pubyear}[1]{\def\@pubyear{#1}}
\newcommand{\show@pubyear}{\@pubyear}
\def\@doi{10.5117/ccr20xx.xxx.xxxx}
\newcommand{\doi}[1]{\def\@doi{#1}}
\newcommand{\show@doi}{\@doi}


\newcommand{\firstpage}[1]{\setcounter{page}{#1}}

%%%%%%%%%%%%%% HEADER / FOOTER %%%%%%%%%%%%%%%%%%%%%%%%
\RequirePackage{lastpage}
\RequirePackage{fancyhdr}
\RequirePackage{graphicx}
\newcommand{\smallcaps}[1]{\textsc{\footnotesize #1}}
\newcommand{\smallcapsl}[1]{\MakeLowercase{\smallcaps{#1}}}
\setlength{\headheight}{46pt}

\fancypagestyle{firstpage}{%
\fancyhf{} % clear all six fields
\fancyhead[L]{\includegraphics[height=4em]{ufabc-logo.jpg}}
\fancyhead[R]{\smallcaps{Curso: Processos de Fabricação de Dispositivos Eletrônicos\\
Profª: Denise Criado Pereira de Souza. pg. 
               \oldstylenums{\thepage}--\oldstylenums{\pageref{LastPage}}\vspace{-.3em}}}

\fancyfoot[L]{\vspace{-.2em}\\\smallcaps{\thepage}}
\renewcommand{\headrulewidth}{1pt} %linha cabeçalho
\renewcommand{\footrulewidth}{1pt} %linha rodapé
}
\fancypagestyle{followingpage}{%
\fancyhf{} % clear all six fields
\fancyfoot[LO,RE]{\smallcaps{\thepage}}
\renewcommand{\headrulewidth}{1Pt} %linha cabeçalho
\renewcommand{\footrulewidth}{1pt} %linha rodapé
}
\pagestyle{followingpage}
\AtBeginDocument{\thispagestyle{firstpage}}
%%%%%%%%%%%%%%%%%% Other markup / styling %%%%%%%%%%%%%%%%%%%%%

% More compact enumerations
\renewcommand{\@listI}{%
  \itemsep=0\parsep}
% Bibliography style

%%%%%%%%%%%%% Author information  black magic %%%%%%%%%%%%%%%%%
% Mostly stolen from https://github.com/dan-weiss/apa7-latex-cls-source/blob/62f31e0b2c8c75e260a7690928c745d803333549/apa7/apa7.dtx (LPPL licensed)

\newcommand*\listauthors{}
\newcommand*\listsuperscripts{}
\newcommand*\listaffiliations{}

\newcommand*{\authorsnames}[2][]{
  \def\def@multipleauthors{\@multipleauthorsmode} %
  \renewcommand*\listauthors{}
  \renewcommand*\listsuperscripts{}
  \newcounter{NumberOfAuthors}
  \newcounter{NumberOfSuperscripts}
  \forcsvlist{\stepcounter{NumberOfAuthors}\listadd\listauthors}{#2}
  \forcsvlist{\stepcounter{NumberOfSuperscripts}\listadd\listsuperscripts}{#1}
  }
  
\newcommand*{\authorsaffiliations}[1]{
  \def\def@multipleaffils{\@multipleaffilsmode} %
  \renewcommand*\listaffiliations{}
  \newcounter{NumberOfAffiliations}
  \forcsvlist{\stepcounter{NumberOfAffiliations}\listadd\listaffiliations}{#1}
  }
  
\catcode`\|=3
  
\def\looptwo#1#2{%
    \edef\tmp{\noexpand\xtwo%
      \unexpanded\expandafter{#1}\relax  % no added delimiter here
      \unexpanded\expandafter{#2}\relax  % no added delimiter here
    } \tmp%
  }%
  
\def\xtwo#1|#2\relax#3|#4\relax{%
    \dotwo{#1}{#3}%
    \def\tmp{#2}%
    \ifx\empty\tmp%
      \expandafter\@gobble%
    \else%
      \expandafter\@firstofone%
    \fi%
    {\xtwo#2\relax#4\relax}%
  }%

  \catcode`\|=12
\newcommand*{\dotwo}[2]{}


%%%%%%%%%%%%%% Title page %%%%%%%%%%%%%%%%%%%%%
\RequirePackage{calc}
\RequirePackage{pstricks}
\RequirePackage{hyphenat}

\renewcommand{\maketitle}{%
\sloppy
\noindent{\fontsize{14}{13.5}\fontseries{b}\selectfont\raggedright\nohyphens{\@title}}
\vspace{.1in}

\fussy
\renewcommand*{\dotwo}[2]{%
\noindent##1\\
\noindent\textit{##2}
\vspace{0em}\par
}
\looptwo\listauthors\listaffiliations

\vspace{0em}
\parbox{\textwidth-\parindent-\parindent}{\small

}
}

%%%%%%%%%%%%%% Sections %%%%%%%%%%%%%%%%%%%%%
\renewcommand{\section}{%
\@startsection{section}{1}{0pt}%
{-1.5ex plus -1ex minus -.2ex}{1.5ex}%
{\fontsize{14}{13.5}\fontseries{b}\selectfont}%
}
\setcounter{secnumdepth}{0}

\renewcommand{\subsection}{%
\@startsection{subsection}{2}{0pt}%
{-1.5ex plus -1ex minus -.2ex}{1.5ex}%
{\fontsize{13}{12.5}\fontseries{}\selectfont}%
}
\setcounter{secnumdepth}{3}


%%%%%%%%%%%%%% Reference handling %%%%%%%%%%%%%%%%%%%%%
\RequirePackage{csquotes}
\RequirePackage[style=apa,sortcites=true,sorting=nyt,backend=biber]{biblatex}

\DeclareLanguageMapping{portuguese}{portuguese-apa}


\renewcommand{\bibfont}{\small}
\setlength{\bibhang}{\parindent}