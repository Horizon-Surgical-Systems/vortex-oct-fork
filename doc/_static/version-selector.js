(function () {
    const version_selector_container = 'sidebar-tree';
    const warning_container = 'sidebar-brand';

    let info;

    function normalize_url(url) {
        if(url.startsWith('/')) {
            url = window.origin + url;
        }
        return url;
    }

    function get_redirect_url(base_target_url, cb) {
        if (info !== undefined && window.location.href.startsWith(info.url)) {

            // map current page to corresponding page in other version
            const target_url = base_target_url + window.location.href.substring(info.url.length);

            // check that destination page exists
            fetch(target_url, { 'method': 'HEAD' })
                .then((response) => {
                    if(response.ok) {
                        // page is valid so report
                        cb(target_url);
                    } else {
                        // page is invalid so use homepage
                        cb(base_target_url);
                    }
                })
                .catch((reason) => {
                    // an error occurred so use homepage
                    cb(base_target_url);
                });

        } else {
            cb(base_target_url);
        }
    }

    function add_version_message(header, type, title, message, url) {
        get_redirect_url(normalize_url(url), (url) => {
            header.insertAdjacentHTML('afterend', `<p class="version-message version-${type}"><span>${title}<br /><a href="${url}">${message}</a></span></p>`);
        });
    };

    fetch('/docver-v0.json')
        .then((response) => response.json())
        .then((data) => {

            // determine if version is active using the Sphinx-provided metadata
            // NOTE: Sphinx is configured to add a leading "v" to the version
            const current_version = DOCUMENTATION_OPTIONS.VERSION.substring(1);
            for (const obj of data) {
                if(obj.version == current_version) {
                    info = obj;
                    break;
                }
            }
            if(info === undefined) {
                console.warn(`did not find current version ${current_version} in database`);
            }

            const container = document.getElementsByClassName(version_selector_container)[0];
            if(container === undefined) {
                console.warn(`version selector container ${version_selector_container} is missing`);
            } else {

                // create version selector dropdown
                const elem = document.createElement('select');
                elem.setAttribute('class', 'version-selector reference');

                for (obj of data) {
                    const o = document.createElement('option');

                    obj.url = normalize_url(obj.url);
                    o.setAttribute('value', obj.url);
                    o.innerText = (obj.name !== undefined) ? obj.name : `v${obj.version}`;
                    if (info !== undefined && obj.version == info.version) {
                        o.setAttribute('selected', 'selected');
                    }

                    elem.appendChild(o);
                }

                // attach event handler to switch documentation versions
                elem.addEventListener('change', () => {
                    get_redirect_url(elem.value, (url) => {
                        window.location.href = url;
                    });
                });

                // insert into table of contents sidebar
                container.insertAdjacentHTML('beforeend', '<p class="caption" role="heading"><span class="caption-text">Version</span></p>');
                let ul = document.createElement('ul');
                let li = document.createElement('li');
                li.setAttribute('class', 'toctree-l1');
                ul.appendChild(li);

                li.appendChild(elem);
                container.append(ul);
            }

            // add warnings
            if (info !== undefined && (info.develop || !info.active)) {
                let header = document.getElementsByClassName(warning_container)[0];
                if(header === undefined) {
                    console.warn(`warning container ${warning_container} is missing`);
                } else if(info.develop) {
                    add_version_message(header, 'develop', 'Development Release', 'Switch to Stable', '/rel/stable/doc');
                } else if (!info.active) {
                    add_version_message(header, 'legacy', 'Legacy Release', 'Switch to Latest', '/rel/latest/doc');
                }
            }
        });
}) ();
