import pylab
import numpy as np
import copy
from tqdm import tqdm

#rupostagger
import rutokenizer
import rupostagger
import rulemma

lemmatizer = rulemma.Lemmatizer()
lemmatizer.load()

tokenizer = rutokenizer.Tokenizer()
tokenizer.load()

tagger = rupostagger.RuPosTagger()
tagger.load()

#natasha
from slovnet import Syntax
from slovnet import Morph
from razdel import sentenize, tokenize
from navec import Navec

np.long = np.integer

navec = Navec.load('navec_news_v1_1B_250K_300d_100q.tar')
syntax = Syntax.load('slovnet_syntax_news_v1.tar')
syntax.navec(navec)
morph = Morph.load('slovnet_morph_news_v1.tar', batch_size=4)
morph.navec(navec)
print(end = '')

class word_token:
    
    def __init__ (self, text, pos, sentence_num = -1, sentence_pos = -1, is_direct = False, is_authors = False, is_actor = False):
        self.text = text
        self.pos = pos
        
        self.sentence_num = sentence_num
        self.sentence_pos = sentence_pos
        
        self.is_direct = False
        self.is_authors = False
        self.is_actor = False
        
    def __repr__ (self):
        return "{} ({})".format(self.text,self.pos)
    
    def __str__ (self):
        return "{} ({})".format(self.text,self.pos)
    
    def __lt__ (self,other):
        if(self.text < other.text):
            return True
        else:
            return False
        
class direct_speach:
    
    def __init__(self,direct_, authors_):
        self.direct = direct_
        self.authors = authors_
        self.position_in_text = -1 #номер предложения в тексте
        self.actor_words = None #слова, используемые в этом предложении при указании на говорящего
        self.actor = None #объект актор, который произносит прямую речь
            
    def __repr__ (self):
        res_str = ''
        for sent in self.direct:
            for word in sent:
                res_str += word.text + ' '
        res_str += ' -- '
        for sent in self.authors:
            for word in sent:
                res_str += word.text + ' '
        res_str += ' -- '
        res_str += "\n \t by {}".format(self.actor_words)
        return res_str
    
    def __str__ (self):
        res_str = ''
        for sent in self.direct:
            for word in sent:
                res_str += word.text + ' '
        res_str += ' -- '
        for sent in self.authors:
            for word in sent:
                res_str += word.text + ' '
        res_str += ' -- '
        res_str += "\n \t by {}".format(self.actor_words)
        return res_str
        
        
class actor_obj:
    
    def __init__(self,names_):
        self.names = names_ #это список
        self.freq = np.zeros(len(names_),dtype = int).tolist()
        self.utterances = []
        self.active = True
        
    def add_utterance(self, utt):
        self.utterances.append(utt)
        for act_token in utt.actor_words: #альтернативные имена актора, которые были в присоединенном предложении
            word = kz_lemma(act_token.text)
            if(pos(word) != 'PRON'):
                if(not word in self.names):
                    self.names.append(word)
                    self.freq.append(1)
                else:
                    self.freq[self.names.index(word)] += 1
        
    def merge(self, other):
        for new_name in other.names:
            if(new_name not in self.names):
                self.names.append(new_name)
                self.freq.append(other.freq[other.names.index(new_name)])
            else:
                self.freq[self.names.index(new_name)] += other.freq[other.names.index(new_name)]
                
        self.utterances += other.utterances
        
        
    def __repr__ (self):
        str = "actor by names\n\t"
        for i in range(len(self.names)):
            str += "{} ({}, ".format(self.names[i],self.freq[i])
        str += "\n"
        str += "\thas {} utterances".format(len(self.utterances))
        return str
    
    def __str__ (self):
        str = "actor by names\n\t"
        for i in range(len(self.names)):
            str += "{} ({}), ".format(self.names[i],self.freq[i])
        str += "\n"
        str += "\thas {} utterances".format(len(self.utterances))
        return str


def pos(word):
    token = tokenizer.tokenize(word)
    tag = [[i] for i in tagger.tag(token)][0][0]
    pos = ''
    for lit in tag[1]:
        if (lit != '|'):
            pos += lit
        else:
            return pos
        
def kz_lemma(word):
    token = tokenizer.tokenize(word)
    tag = tagger.tag(token)
    lemma = lemmatizer.lemmatize(tag)
    return lemma[0][2]

def kz_lemma_token(word_token):
    new_token = copy.deepcopy(word_token)
    new_token.text = kz_lemma(word_token.text)
    return new_token

def case(word):
    token = tokenizer.tokenize(word)
    tag = [[i] for i in tagger.tag(token)][0][0]
    
    at = False
    rd = False
    case = ''
    for lit in tag[1]:
        if (lit == '|'):
            if (not at):
                at = True
                continue
            else:
                return case
            
        if(lit == '=' and at == True):
            rd = True
            continue
        
        if (rd):
            case += lit
            
def tokenise(word):
    token = tokenizer.tokenize(word)
    tag = [[i] for i in tagger.tag(token)][0][0]
    return tag
    
            
def case_ok_kz(word):
    if(case(word) in ['Nom','Acc']):#acc для Берлиоза, потому что он его так определяет
        return True
    return False

def mp(token): 
    return next(morph.map([[token]])).tokens[0]

def is_not_anim_shurely(token):
    if('Animacy' in token.feats.keys() and token.feats['Animacy'] in ['Inan']):
        return True
    else:
        return False
    
def case_ok(token):
    if('Case' in token.feats.keys() and token.feats['Case'] in ['Nom','Gen']):
        return True
    else:
        return False


def sentence_parce_tkn(sentence, edit_plain_document = False, plain_document = None, plane_document_sentence_start = None):
    in_sent_div_pred = ['',':',',','!','?','.','...','?..','!..','»'] #' ' в предразделителях определяет режим работы 
    in_sent_div = ['–']
    
    if(sentence[0].text == '–'):
        #sentences.append(sentence)
        direct_array = []
        authors_array = []
        
        direct = [sentence[0]]
        author = []
        state = 'direct'
        switch = 'off'
        buf = ''
        
        word_num = 0
        for word in sentence[1:]:
            word_num += 1
            #switch rules go first
            
            if(word.text in in_sent_div_pred):
                switch = 'armed'
            
            if(word.text in in_sent_div and switch == 'armed'):
                direct.append(word)
                
                switch = 'off'
                
                if(state == 'direct'):
                    state = 'authors'
                    direct_array.append(direct)
                    direct = []
                    continue
                    
                if(state == 'authors'):
                    authors_array.append(author)
                    author = []
                    state = 'direct'
                    continue
                    
            if(switch == 'armed' and not word.text in in_sent_div_pred):
                
                switch = 'off'
                
                        
            if(state == 'direct'):
                if(edit_plain_document):
                    plain_document[plane_document_sentence_start + word_num].is_direct = True
                direct.append(word)
                
            if(state == 'authors'):
                if(edit_plain_document):
                    plain_document[plane_document_sentence_start + word_num].is_authors = True
                author.append(word)
                
        if(state == 'direct'):
            direct_array.append(direct)
        if(state == 'authors'):
            authors_array.append(author)
            ""
        if(edit_plain_document):
            return direct_speach(direct_array,authors_array), plain_document
            
        return (direct_speach(direct_array,authors_array))
    
    if(edit_plain_document):
        return None, plain_document
    
    return None


def extract_actors(utterances = None,  syntax = syntax, morph = morph, 
                  use_methods = ["main","propn","naive","res_adv","res"],verbal = True):
    
    TSP = []
    stats = [0,0,0,0,0]

    for utterance in utterances:
        if(utterance.authors == ''):
            TSP.append([utterance,[],[],[],[]])
            continue

        p1 = []
        p2 = []
        p3 = []
        p4 = []
        burnout = 3 #сколько слов назад видит запасный метод выделения актора
        
        direct_parts = utterance.direct
        parts = utterance.authors
        parts_cnt = 0

        past_parts_len = 0
        for part in parts:
            was_comma = False
            
            direct_tokens = direct_parts[parts_cnt]
            past_parts_len += len(direct_tokens)
            
            tokens = [_.text for _ in part]
            tokens_s = next(syntax.map([tokens])).tokens
            tokens_m = next(morph.map([tokens])).tokens

            if(verbal):
                print("-----------------------------------------")
                

            for i in range(len(tokens)):

                if(tokens_s[i].text == ','):
                    #чтобы случайно не обрубить по союзу, что не соединяет предложения
                    was_comma = True
                    if(verbal):
                        print('{}'.format(tokens_s[i].text),end=' ')
                    continue

                if((tokens_s[i].rel == "cc" and was_comma) or tokens_s[i].text == "."): 
                    # обрубаем вторую часть сложных предложений, потому что она не нужна и путает парсер
                    was_comma = False
                    break
                    
                was_comma = False
                
                #основной метод: root + nsubj
                    
                if("main" in use_methods):
                    if(tokens_s[i].rel == 'nsubj' and pos(tokens_m[i].text) in ['NOUN','ADJ','PRON']):
                        if(tokens_s[int(tokens_s[i].head_id) - 1].rel in ['root']):
                            if(verbal):
                                print("\033[7m{}\033[0m".format(tokens_s[i].text),end=' ')
                            #p1.append(word_token(tokens_s[i].text,i + past_parts_len, is_actor = True))
                            #если помечать слова, как акторы, здесь, то много лишних слов будет помечено, как акторы
                            #part[i].is_actor = True
                            p1.append(part[i])
                            continue

                #дополнительный метод: имя собственное

                if("propn" in use_methods):
                    if((tokens_m[i].pos) == 'PROPN' and case_ok_kz(tokens_m[i].text) and not is_not_anim_shurely(tokens_m[i])):
                        if(verbal):
                            print("\033[37m\033[7m{}\033[0m".format(tokens_s[i].text),end=' ')
                        #part[i].is_actor = True
                        p2.append(part[i])
                        #p2.append(word_token(tokens_s[i].text,i + past_parts_len, is_actor = True))
                        continue

                #резервный метод: относящееся к глаголу слово, часть речи которого не попадает в список запрещенных

                if("res_adv" in use_methods):
                    if(pos(tokens_s[int(tokens_s[i].head_id) - 1].text) == 'VERB'): #которые относятся к глаголу
                        if(pos(tokens_m[i].text) in ['NOUN','ADJ','PRON'] and case_ok_kz(tokens_m[i].text)):
                            if(verbal):
                                print("\033[35m{}\033[0m".format(tokens_s[i].text),end=' ')
                            #part[i].is_actor = True
                            p3.append(part[i])
                            #p3.append(word_token(tokens_s[i].text,i + past_parts_len, is_actor = True))
                            continue

                #резервный метод: имя
                #насколько стоит смотреть на падежи хз, так правильно, но точность чуть снижается
                if("res" in use_methods):
                    if(pos(tokens_m[i].text) in ['NOUN','ADJ','PRON'] and case_ok_kz(tokens_m[i].text)):
                        if(verbal):
                            print("\033[37m{}\033[0m".format(tokens_s[i].text),end=' ')
                            
                        #part[i].is_actor = True
                        p4.append(part[i])
                        #p4.append(word_token(tokens_s[i].text,i + past_parts_len, is_actor = True))
                        continue
                        
                        
                if(tokens_s[i].head_id == '0'): 
                    #на root тоже надо смотреть, так как наташа может выделить не рут как рут
                    if(verbal):
                        print("\033[34m{}\033[0m".format(tokens_s[i].text),end=' ')
                    continue

                if(verbal):
                    if(tokens_s[i].rel == "cc"):
                        print('\033[31m{}\033[0m'.format(tokens_s[i].text),end=' ')
                    else:
                        print('{}'.format(tokens_s[i].text),end=' ')
                        
                
                    
                    
            if(verbal):
                print()
                print()
                print(p1)
                print(p2)
                print(p3)
                print(p4)
                print()

                print("--------------")

            for i in range(len(tokens)):
                if(verbal):
                    print("\t{}\n\t{}\n".format(tokens_s[i],tokens_m[i]))
                    
            past_parts_len += len(tokens)
            parts_cnt += 1

        TSP.append([utterance,p1,p2,p3,p4])

        stats[0] += 1
        if(p1 != []):
            stats[1] += 1
        else:
            if(p2 != []):
                stats[2] += 1
            else:
                if(p3 != []):
                    stats[3] += 1
                else:
                    if(p4 != []):
                        stats[4] += 1





    return TSP,stats  



def format_result(TSP,verbal = True):
    resut = []
    reserve = []
    for ut_n in range(len(TSP)):
        res_prime = []
        res_scond = []
        
        found = 0
        
        for act in TSP[ut_n][1]:
            found = 1
            res_prime.append(act)
            
        if(not found):
            for act in TSP[ut_n][2]:
                res_prime.append(act)
                found = 2
                
        if(found == 1):
            for act in TSP[ut_n][2]:
                res_scond.append(act)
                
        if(not found):
            for act in TSP[ut_n][3]:
                res_prime.append(act)
                found = 3
                
        if(found == 2):
            for act in TSP[ut_n][3]:
                res_scond.append(act)
                
        if(not found):
            for act in TSP[ut_n][4]:
                res_prime.append(act)        
            
        resut.append(res_prime)
        reserve.append(res_scond)
        
    return resut, reserve


def merge_results(main,reserve):
    result = []
    for i in range(len(main)):
        result.append(main[i] + reserve[i])
    return result

def get_sentence(text, num):
    res = []
    for token in text: 
        if(token.sentence_num == num):
            res.append(token)
    return res


def find_actors_in_last_part_of_a_sentence(sentence):
    was_cc = False
    tokenised_sent = []
    tokens = [_.text for _ in sentence]
    tokens_s = next(syntax.map([tokens])).tokens
    for i in range(len(tokens) - 1, -1, -1):
        if(tokens_s[i].rel == 'cc'):
            was_cc = True
            continue

        if(tokens_s[i].rel == 'punct' and was_cc):
            break

        was_cc = False
        tokenised_sent.append(sentence[i])

    tokenised_sent.reverse()

    TSt = extract_actors(utterances = [direct_speach([[word_token('none',None)]],[tokenised_sent])], verbal = False)
    resut,reserve = format_result(TSt[0],verbal = False)
    result = merge_results(resut,reserve)
    return result[0]


def in_a_same_pair(lhs, rhs, coreferent_pairs):
    for pair in coreferent_pairs:
        if(lhs in pair and rhs in pair):
            return True
    return False

def find_name_for_pron(tokenised_plain_text, possible_replacements, pron_sent, pron_sent_pos, backsearch_limit = 50):
    
    for i in range(len(tokenised_plain_text)):
        # поиск позиции метоимения, что надо заменить
        if(tokenised_plain_text[i].sentence_num == pron_sent):
            
            if(tokenised_plain_text[i].sentence_pos == pron_sent_pos):
                # нашли
                # Проверим, что рил местоимение
                if(pos(kz_lemma(tokenised_plain_text[i].text)) == 'PRON'):

                    # идем назад
                    for j in range(max(i - 1, 0), max(i - backsearch_limit, 0), -1):
                        #print("|{}| ".format(kz_lemma(tokenised_plain_text[j].text)),end = '')

                        if(kz_lemma(tokenised_plain_text[j].text) in possible_replacements):
                            
                            if(pos(kz_lemma(tokenised_plain_text[j].text)) != 'PRON'):

                                if(compatible(kz_lemma(tokenised_plain_text[i].text), kz_lemma(tokenised_plain_text[j].text))):
                                    #нашли имя, которое заменяет местоимение
                                    token_copy = copy.deepcopy(tokenised_plain_text[j])
                                    token_copy.text = kz_lemma(token_copy.text)
                                    return token_copy
    return None

def format_tags(word):
    tags = []
    vals = []
    token = tokenise(word)[1]
    
    tag = ''
    val = ''
    rdtag = True
    for liter in token:
        if(liter == '='):
            rdtag = False
            continue
            
        if(liter == '|'):
            if(val == ''):
                tag = ''
                val = ''
                continue
            tags.append(tag)
            vals.append(val)
            rdtag = True
            
            tag = ''
            val = ''
            continue
            
        
        if(rdtag == True):
            tag += liter
            
        if(rdtag == False):
            val += liter
        
    
    tags.append(tag)
    vals.append(val)
    
    return tags,vals



def gender(word):
    tags,vals = format_tags(word)
    
    i = 0
    for tag in tags:
        if(tag == 'Gender'):
            return vals[i]
        
        i += 1
        
    return None

def number(word):
    tags,vals = format_tags(word)
    
    i = 0
    for tag in tags:
        if(tag == 'Number'):
            return vals[i]
        
        i += 1
        
    return None

def anim(token):
    if('Animacy' in token.feats.keys()):
        if(token.feats['Animacy'] == 'Inan'):
            return False
        if(token.feats['Animacy'] == 'Anim'):
            return True
        
    return None



def compatible(lhs,rhs,verbal = False):
    #сравнивать по одушевленности
    #по роду
    #мб по числу
    
    lhs_tkn = [_.text for _ in tokenize(lhs)]
    rhs_tkn = [_.text for _ in tokenize(rhs)]
    token_lhs = next(morph.map([lhs_tkn])).tokens[0]
    token_rhs = next(morph.map([rhs_tkn])).tokens[0]
    
    if(verbal):
        print("\t{}\t{}".format(gender(lhs),gender(rhs)))
        print("\t{}\t{}".format(number(lhs),number(rhs)))
        print("\t{}\t{}".format(anim(token_lhs),anim(token_rhs)))
    
    if(gender(lhs) != gender(rhs)):
        return False
    
    if(number(lhs) != number(rhs)):
        return False
    
    return True   

    if(anim(token_lhs) == True and anim(token_rhs) == False):
        return False
    
    if(anim(token_rhs) == True and anim(token_lhs) == False):
        return False
     

class text_parser:
    sentencised_text = None
    plain_tokenised_text = None
    utterances = None
    extracted_actor_words = None
    referents = None
    
    def __init__ (self, text):
        self.sentencised_text = text
        return
    
    def extract_utterances_and_tokenise(self):
        # ТРЕБУЕТ sentencised_text
        # ЗАПОЛНЯЕТ plain_tokenised_text, utterances
        
        tokenised_plain_text = []

        utterances = []

        past_len = 0
        num = 0
        for sentence in self.sentencised_text:
            tokenised_sentence = [word_token(_.text, -1, sentence_num = num, sentence_pos = -1) for _ in tokenize(sentence)]

            for i in range(len(tokenised_sentence)):
                tokenised_sentence[i].pos = i + past_len
                tokenised_sentence[i].sentence_pos = i

            prev_len = len(tokenised_plain_text)
            tokenised_plain_text += tokenised_sentence

            res, tokenised_plain_text = sentence_parce_tkn(tokenised_sentence, edit_plain_document = True, plain_document = tokenised_plain_text, plane_document_sentence_start = prev_len)
            
            if(res):
                res.position_in_text = num
                utterances.append(res)
                res.text = sentence


            num += 1
            past_len += len(tokenised_sentence)
        
        self.plain_tokenised_text = tokenised_plain_text
        self.utterances = utterances
        
        return
    
    def extract_actor_words_from_utterances(self):
        # 
        
        TSP,stats = extract_actors(utterances = self.utterances, syntax = syntax, morph = morph, verbal = False, use_methods = ["main","propn","naive","res_adv","res"])
        resut,reserve = format_result(TSP,verbal = False)
        result = merge_results(resut,reserve)
        self.extracted_actor_words = result
        
        for i in range(len(self.utterances)):
            self.utterances[i].actor_words = result[i]
    
    def construct_actor_objects(self):
        
        # формирование списков слов-акторов
        extracted_words = []
        for element in self.extracted_actor_words:
            for word in element:
                if(pos(word.text) != 'PROPN'):
                    extracted_words.append(word.text.lower())
                else:
                    extracted_words.append(word.text)
        
        unique_extracts, counts = np.unique(extracted_words,return_counts = True)
        unique_extracts_s = unique_extracts[np.argsort(-counts)]
        counts_s = counts[np.argsort(-counts)]
        unique_count_pairs = np.array([[unique_extracts_s[i], counts_s[i]] for i in range(len(unique_extracts))])
        
        ashured_actor_words = []
        unique_actor_words = []
        for element in unique_count_pairs:
            if(int(element[1]) > 1):
                ashured_actor_words.append(kz_lemma(element[0]))
            unique_actor_words.append(kz_lemma(element[0]))
            
        # поиск соответствующего актора для каждого предложения
        
        # 1. поиск кореферентных пар самым наивным образом
        coreferent_pairs = []

        backsearch_limit = 100

        cur_sent = 0
        block_last = ""
        block_case = ""
        for i in range(len(self.plain_tokenised_text)):

            if(pos(kz_lemma(self.plain_tokenised_text[i].text)) != 'PRON'):
                if(kz_lemma(self.plain_tokenised_text[i].text) in unique_actor_words):
                    if(block_last != ""):
                        # валидное слово-актор
                        if(block_case == case(self.plain_tokenised_text[i].text) and compatible(self.plain_tokenised_text[i].text, block_last)):
                            coreferent_pairs.append([kz_lemma(block_last),kz_lemma(self.plain_tokenised_text[i].text)])


                    block_last = self.plain_tokenised_text[i].text
                    block_case = case(self.plain_tokenised_text[i].text)

            if(self.plain_tokenised_text[i].text in [',','.','!','?',';','и','!..','?..','–','«','»']):
                block_last = ""
                block_case = ""      
        
        # слова, которые попали в кореферентные пары можно добавить к словам, которые точно указывают на актора
        for pair in coreferent_pairs:
            for word in pair:
                if (not word in ashured_actor_words):
                    ashured_actor_words.append(word)  
                
        # 2. поиск соответствующих акторов
        
        actors_objects = []
        for actor_word in ashured_actor_words:
            if(pos(actor_word) != 'PRON'):
                act = actor_obj([actor_word])
                actors_objects.append(act)

        # можно либо пройти один раз по всем высказыванием, дать высказывание одному актору, а затем сливать синонимичных акторов
        # либо для каждого актора отобрать предложения, которые ему подходят, а затем удалить дубликаты
        utterances_overall = 0.
        utterances_solved = 0.

        utt_num = 0
        for utterance in self.utterances: #проход по предложениям и определение соотв. акторов

            # если в предложении только одно слово, обозначающее актора, и это местоимение, то только по нему ничего не определить.
            # надо найти имя, которое это местоимение замещает
            valid_actor_words = []

            # если слов - акторов в предложении не нашли
            if(len(utterance.actor_words) == 0):
                # сначала проверим предыдущее предложение
                prev_sentence = get_sentence(self.plain_tokenised_text,max(utterance.position_in_text - 1, 0 ))
                if(prev_sentence[len(prev_sentence) - 1].text == ':'):
                    # случай, когда авторская речь помещена в прошлом предложении
                    # бывает, что на актора указывают в нем
                    prev_sent_actor_words = find_actors_in_last_part_of_a_sentence(prev_sentence)
                    if(len(prev_sent_actor_words) != 0):
                        # нашли слова-акторы

                        utterance.actor_words = prev_sent_actor_words


            # обработка местоимений и перенос слов-акторов в промежуточный массив + лемматизация слов-акторов
            only_pron = True
            for actor_word in utterance.actor_words: #весь текст в utterance.actor_words лемменизирован заранее
                if(pos(actor_word.text) != 'PRON'):
                    valid_actor_words.append(kz_lemma_token(actor_word)) # все не местоимения считаются валидными 
                    if(kz_lemma(actor_word.text) in ashured_actor_words): # но если слово не в ashured actor words то местоимения все равно надо заменить на имена
                        only_pron = False

            #если в предложении не только местоимения, то лучше не засорять слова-акторы лишними именами, так как поиск замены местоимения может ошибиться

            if(only_pron):
                for actor_word in utterance.actor_words:
                    replacement = find_name_for_pron(self.plain_tokenised_text, ashured_actor_words, utterance.position_in_text, actor_word.sentence_pos, backsearch_limit)
                    # find_name_for_pron берет замену из текста и лемменизирует токен
                    if(replacement != None):
                        valid_actor_words.append(kz_lemma_token(replacement))


            # местоимения заменены, теперь можно искать подходящего актора

            this = False
            for actor in actors_objects:
                for word in valid_actor_words:
                    if(kz_lemma(word.text) in actor.names):
                        actor.add_utterance(utterance)
                        utterance.actor = actor
                        this = True
                        break
                if(this): # каждое предложение относится только к одному актору. Как только актор найден поиск останавливается
                    break

            if(not this):
                # не нашли актора, применяем метод чередования
                if(len(utterance.actor_words) == 0):
                    # в случае если слов, указывающих на актора нет, считаем, что имеет место диалог между двумя акторами и подразумевается чередование фраз (иначе читатель ничего не поймет)
                    if(self.utterances[utt_num - 2].actor != None):
                        actors_objects[actors_objects.index(self.utterances[utt_num - 2].actor)].add_utterance(utterance)
                        utterance.actor = self.utterances[utt_num - 2].actor



            utt_num += 1

        #for utterance in self.utterances:
        #    utterances_overall += 1
        #    if(utterance.actor == None):
        #        print(utterance)
        #        print("-----------------")
        #    else:
        #        utterances_solved += 1

        for actor in actors_objects: # слияние синонимичных акторов, отключение дубликатов
            if(actor.active):
                for other_actor in actors_objects:
                    if(other_actor.active):
                        if(other_actor != actor):
                            for actor_name in actor.names:
                                if (actor_name in other_actor.names):
                                    actor.merge(other_actor)
                                    #actors_objects.remove(other_actor)
                                    other_actor.active = False
                                    break

                                for other_name in other_actor.names:
                                    if(in_a_same_pair(actor_name, other_name, coreferent_pairs)):
                                        actor.merge(other_actor)
                                        #actors_objects.remove(other_actor)
                                        other_actor.active = False
                                        break


        #print("-------------------------------------------")
        #print("found actor for {} utterances".format(utterances_solved / utterances_overall))
        
        self.referents = actors_objects

        return
    
    def print_text_formated(self):
        cur_sent = 0
        
        for i in range(len(self.plain_tokenised_text)):
            if(self.plain_tokenised_text[i].sentence_num != cur_sent):
                print()
                cur_sent += 1

            if(self.plain_tokenised_text[i].is_actor):
                print("\033[33m{}\033[0m".format(self.plain_tokenised_text[i].text),end=' ')
                continue
            if(self.plain_tokenised_text[i].is_authors):
                print("\033[32m{}\033[0m".format(self.plain_tokenised_text[i].text),end=' ')
                continue
            if(self.plain_tokenised_text[i].is_direct):
                print("\033[34m{}\033[0m".format(self.plain_tokenised_text[i].text),end=' ')
                continue

            print(self.plain_tokenised_text[i].text,end=' ')
    
    def format_actors_in_text(self):
        
        for utterance in self.utterances:
            for i in range(len(self.plain_tokenised_text)):

                if(self.plain_tokenised_text[i].sentence_num in [_.sentence_num for _ in utterance.actor_words]):
                    if(self.plain_tokenised_text[i].sentence_pos in [_.sentence_pos for _ in utterance.actor_words]):
                        self.plain_tokenised_text[i].is_actor = True

            
def import_text(fname):
    document = []

    dividers = ['\n']
    skip = ['\n']

    sent = ''
    fin = open(fname,'r',encoding="utf8")
    letter = fin.read(1)
    while (letter != ''):
        if(letter in dividers):
            if(sent != ''):
                document.append(sent)
            sent = ''
            letter = fin.read(1)
            letter = fin.read(1)
            continue
        if(letter in skip):
            letter = fin.read(1)
            continue
        sent += letter

        letter = fin.read(1)
    fin.close()
    
    return document





