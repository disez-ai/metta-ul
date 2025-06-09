(: Lambda (-> Atom $t (-> $a $t)))
(=
    ((Lambda $var $body) $arg)
    (let $var $arg $body)
)