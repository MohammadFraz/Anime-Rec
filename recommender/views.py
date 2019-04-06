from django.shortcuts import render
from urllib.request import urlopen
from recommender.models import Anime, Genre
import json
from sklearn import linear_model
import numpy as np


def index(request):
    return render(request, 'recommender/index.html')

def credits(request):
    return render(request, 'recommender/credits.html')


def recommender(request):
    try:
        with urlopen('https://myanimelist.net/animelist/' + str(request.POST.get('username')) + '/load.json?status=7&offset=0') as url:
            anime_list = json.loads(url.read().decode())
    except:
        err = 'Invalid username!'
        return render(request, 'recommender/list.html', {'userid': request.POST['username'], 'error': err})
    data_from_users = []
    genre_list = []
    user_score_list = []
    list_watched = []

    for ani in anime_list:
        if ani['score'] != '0':
            try:
                obj = Anime.objects.get(aid=int(ani['anime_id']))
            except Anime.DoesNotExist:
                continue
            list_watched.append(obj.aid)
            genre_one_hot = [0] * 43

            for g in obj.genre.all():
                genre_one_hot[g.gid - 1] = 1

            genre_list.append(genre_one_hot)
            data_from_users.append([float(obj.rating), obj.members])
            user_score_list.append(float(ani['score']))
        elif ani['num_watched_episodes'] != '0':
            list_watched.append(int(ani['anime_id']))

    if len(user_score_list) == 0:
        err = 'No recommendations can be generated since you haven\'t rated any anime.'
        return render(request, 'recommender/list.html', {'userid': request.POST['username'], 'error': err})

    data_from_users = np.array(data_from_users, dtype=float)
    genre_list = np.array(genre_list, dtype=float)
    user_score_list = np.array(user_score_list, dtype=float)

    clf = linear_model.ElasticNet(alpha=0.2)
    clf.fit(genre_list, user_score_list)
    coeff = [idx for idx, x in enumerate(clf.coef_) if abs(x) >= 0.1]

    clf2 = linear_model.LinearRegression()
    clf2.fit(np.hstack((data_from_users, genre_list[:, coeff])), user_score_list)
    # print(np.hstack((data_from_users, genre_list[:, coeff])))
    recommendations = []
    for x in Anime.objects.all():
        if x.aid in list_watched or x.members < 100:
            continue
        genre_one_hot = [0] * 43
        for g in x.genre.all():
            genre_one_hot[g.gid - 1] = 1
        genre_one_hot = np.array(genre_one_hot)
        gl = genre_one_hot[coeff]
        # gl = []
        # for i in coeff:
        #     j = 1
        #     for k in feature_detail[i]:
        #         j *= genre_one_hot[k]
        #     gl.append(j)
        data_from_users = [x.rating, x.members]
        data_from_users = np.array(data_from_users)
        predicted_rating = clf2.predict([np.hstack((data_from_users, gl))])[0]
        recommendations.append([x.aid, x.name, predicted_rating])

    recommendations = sorted(recommendations, key=lambda v: v[2], reverse=True)
    c = 0
    i = -1
    recc_id = []
    reccs = []
    while c < 50:
        i += 1
        obj = Anime.objects.get(aid=recommendations[i][0])
        if len(set(recc_id).intersection([x.aid for x in obj.related.all()])) >= 1:
            print(obj.name)
            continue
        recc_id.append(recommendations[i][0])
        reccs.append(obj)
        c += 1
    # print(recommendations[0], clf2.coef_)
    # l = clf.coef_
    # for i in range(1, 44):
    #     print(Genre.objects.get(gid=i).name, l[i - 1])
    # x = 43
    # for i in range(1, 44):
    #     print(Genre.objects.get(gid=i).name, end=': ')
    #     for j in range(i, 43):
    #         print(Genre.objects.get(gid=j).name + ' ', l[x], end=' ')
    #         x += 1
    #     print('')
    return render(request, 'recommender/list.html', {'userid': request.POST.get('username'), 'recc': reccs})
