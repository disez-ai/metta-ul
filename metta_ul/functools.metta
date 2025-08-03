(: Lambda (-> Atom Atom (-> $a $t)))
(= 
    ((Lambda $var $body) $val)
    (let (quote ($v $b)) (sealed ($var) (quote ($var $body)))
        (let (quote $v) (quote $val) $b)
    )
)