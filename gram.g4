grammar gram;

program: defstates defactions transitions EOF;

defstates: STATES defstate (',' defstate)* ';';
defactions: ACTIONS ID+ (',' ID)* ';';

transitions: trans (trans)*;

defstate: ID (':' INT)?;

trans: transact | transnoact;

transact: ID '[' ID ']' FLECHE INT ':' ID ('+' INT ':' ID)* ';';
transnoact: ID FLECHE INT ':' ID ('+' INT ':' ID)* ';';

STATES: 'States';
ACTIONS: 'Actions';
TRANSITION: 'transition';
DPOINT: ':';
FLECHE: '->';
SEMI: ';';
VIRG: ',';
PLUS: '+';
LCROCH: '[';
RCROCH: ']';

INT: [0-9]+;
ID: [a-zA-Z_][a-zA-Z_0-9]*;
WS: [ \t\n\r\f]+ -> skip;







